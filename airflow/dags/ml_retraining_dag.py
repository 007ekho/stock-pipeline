from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.external_task import ExternalTaskSensor
from docker.types import Mount
from datetime import datetime, timedelta
import os
import boto3
import json

default_args = {
    "owner": "crypto-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def validate_processed_data(**context):
    """Check that processed data exists for last 90 days."""
    config = get_secret(os.environ["SECRET_AIRFLOW"])
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = config["S3_BUCKET"]

    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix="processed/daily/"
    )
    count = response.get("KeyCount", 0)
    if count < 1:
        raise ValueError("No processed data found — run daily processing first")
    print(f"Found {count} processed files for training")


def notify_success(**context):
    print(f"ML retraining completed successfully at {context['ds']}")


with DAG(
    dag_id="ml_retraining",
    default_args=default_args,
    description="LSTM model retraining every 3 days",
    schedule_interval="0 2 */3 * *",  # 2am every 3 days — after daily_processing finishes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["crypto", "ml"],
) as dag:

    # Wait for daily_processing to complete before starting
    wait_for_daily = ExternalTaskSensor(
        task_id="wait_for_daily_processing",
        external_dag_id="daily_processing",
        external_task_id="notify_success",
        allowed_states=["success"],
        execution_delta=timedelta(hours=2),  # daily_processing runs at midnight, we run at 2am
        timeout=7200,  # wait up to 2 hours
        poke_interval=60,  # check every minute
        mode="reschedule",  # release the worker slot while waiting
    )

    validate = PythonOperator(
        task_id="validate_processed_data",
        python_callable=validate_processed_data,
    )

    prepare = SparkSubmitOperator(
        task_id="prepare_training_data",
        application="/opt/airflow/spark/jobs/prepare_training.py",
        conn_id="spark_default",
        application_args=["--days", "90"],
        name="prepare_training_{{ ds }}",
    )

    train = DockerOperator(
        task_id="train_lstm",
        image="stock-pipeline-ml",
        command="python train.py",
        environment={"AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"], "SECRET_ML": os.environ["SECRET_ML"]},
        mounts=[
            Mount(
                source=f"{os.environ['HOST_HOME']}/.aws",
                target="/root/.aws",
                type="bind",
                read_only=True,
            )
        ],
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
    )

    notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    wait_for_daily >> validate >> prepare >> train >> notify