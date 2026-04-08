from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import os
import boto3
import json

default_args = {
    "owner": "crypto-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def validate_raw_data(**context):
    """Check that raw S3 files arrived for today."""
    config = get_secret(os.environ["SECRET_AIRFLOW"])
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = config["S3_BUCKET"]
    date = context["ds"]  # YYYY-MM-DD from Airflow context
    year, month, day = date.split("-")

    topics = ["crypto-prices", "crypto-orderbook"]
    for topic in topics:
        prefix = f"raw/{topic}/{year}/{month}/{day}/"
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        count = response.get("KeyCount", 0)
        if count == 0:
            raise ValueError(f"No files found for {topic} on {date}")
        print(f"Validated {count} files for {topic} on {date}")


def notify_success(**context):
    """Log pipeline success."""
    print(f"Daily processing pipeline completed successfully for {context['ds']}")


with DAG(
    dag_id="daily_processing",
    default_args=default_args,
    description="Daily crypto data processing pipeline",
    schedule_interval="0 0 * * *",  # midnight daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["crypto", "processing"],
) as dag:

    validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    merge = SparkSubmitOperator(
    task_id="merge_raw_files",
    application="/opt/airflow/spark/jobs/merge_raw.py",
    conn_id="spark_default",
    application_args=["--date", "{{ ds }}"],
    name="merge_raw_{{ ds }}",
    )

    join = SparkSubmitOperator(
        task_id="join_topics",
        application="/opt/airflow/spark/jobs/join_topics.py",
        conn_id="spark_default",
        application_args=["--date", "{{ ds }}"],
        name="join_topics_{{ ds }}",
    )

    features = SparkSubmitOperator(
        task_id="feature_engineering",
        application="/opt/airflow/spark/jobs/feature_engineering.py",
        conn_id="spark_default",
        application_args=["--date", "{{ ds }}"],
        name="feature_engineering_{{ ds }}",
    )
    # merge = SparkSubmitOperator(
    #     task_id="merge_raw_files",
    #     application="/opt/airflow/spark/jobs/merge_raw.py",
    #     conn_id="spark_default",
    #     application_args=["--date", "{{ ds }}"],
    #     name="merge_raw_{{ ds }}",
    # )

    # join = SparkSubmitOperator(
    #     task_id="join_topics",
    #     application="/opt/airflow/spark/jobs/join_topics.py",
    #     conn_id="spark_default",
    #     application_args=["--date", "{{ ds }}"],
    #     name="join_topics_{{ ds }}",
    # )

    # features = SparkSubmitOperator(
    #     task_id="feature_engineering",
    #     application="/opt/airflow/spark/jobs/feature_engineering.py",
    #     conn_id="spark_default",
    #     application_args=["--date", "{{ ds }}"],
    #     name="feature_engineering_{{ ds }}",
    # )

    notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    # Task order
    validate >> merge >> join >> features >> notify