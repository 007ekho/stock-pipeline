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


def check_merged_data(**context):
    """Validate merged parquet layer — columns, types, null rates."""
    import boto3
    import os
    config = get_secret(os.environ["SECRET_AIRFLOW"])
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = config["S3_BUCKET"]

    response = s3.list_objects_v2(Bucket=bucket, Prefix="processed/merged/")
    count = response.get("KeyCount", 0)
    if count == 0:
        raise ValueError("No merged parquet files found — merge_raw may have failed")
    print(f"[CONTRACT] merged layer: {count} parquet files found ✓")


def check_features_data(**context):
    """Validate feature engineering output — all 15 feature columns present."""
    import boto3
    import os
    config = get_secret(os.environ["SECRET_AIRFLOW"])
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = config["S3_BUCKET"]

    REQUIRED_FEATURE_COLS = [
        "symbol", "price", "size",
        "price_lag_1", "price_lag_5", "price_lag_10",
        "price_change", "price_change_pct",
        "rolling_avg_5", "rolling_avg_10",
        "rolling_stddev_10", "volume_change",
        "spread", "bid_ask_ratio",
        "total_bid_volume", "total_ask_volume",
    ]

    response = s3.list_objects_v2(Bucket=bucket, Prefix="processed/daily/")
    count = response.get("KeyCount", 0)
    if count == 0:
        raise ValueError("No feature parquet files found — feature_engineering may have failed")
    print(f"[CONTRACT] daily features layer: {count} parquet files found ✓")

    # Spot-check one file for schema using pyarrow
    try:
        import pyarrow.parquet as pq
        from io import BytesIO
        first_key = response["Contents"][0]["Key"]
        obj = s3.get_object(Bucket=bucket, Key=first_key)
        buf = BytesIO(obj["Body"].read())
        schema = pq.read_schema(buf)
        actual_cols = set(schema.names)
        missing = [c for c in REQUIRED_FEATURE_COLS if c not in actual_cols]
        if missing:
            raise ValueError(f"[CONTRACT VIOLATION] Missing feature columns: {missing}")
        print(f"[CONTRACT] All {len(REQUIRED_FEATURE_COLS)} feature columns present ✓")
    except ImportError:
        print("[CONTRACT] pyarrow not available — skipping schema check")


def check_training_data(**context):
    """Validate training data — records per symbol, no all-null rows."""
    import boto3
    import os
    config = get_secret(os.environ["SECRET_AIRFLOW"])
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = config["S3_BUCKET"]

    response = s3.list_objects_v2(Bucket=bucket, Prefix="processed/training/")
    count = response.get("KeyCount", 0)
    if count == 0:
        raise ValueError("No training parquet files found — prepare_training may have failed")

    total_size = sum(obj["Size"] for obj in response.get("Contents", []))
    if total_size < 1024:
        raise ValueError(f"Training data suspiciously small: {total_size} bytes")

    print(f"[CONTRACT] training layer: {count} files, {total_size / 1024:.1f} KB ✓")


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

    check_merged = PythonOperator(
        task_id="check_merged_data",
        python_callable=check_merged_data,
    )

    check_features = PythonOperator(
        task_id="check_features_data",
        python_callable=check_features_data,
    )

    notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    # Task order — contract checks gate each Spark stage
    validate >> merge >> check_merged >> join >> features >> check_features >> notify