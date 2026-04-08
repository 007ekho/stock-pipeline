# import json
# import boto3
# import logging
# import os
# from kafka import KafkaConsumer
# from datetime import datetime, timezone

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # --- Secrets ---
# def get_secret(secret_name: str) -> dict:
#     client = boto3.client("secretsmanager", region_name="eu-north-1")
#     response = client.get_secret_value(SecretId=secret_name)
#     return json.loads(response["SecretString"])


# config = get_secret("stock-pipeline/consumer")
# KAFKA_BOOTSTRAP_SERVERS = config["KAFKA_BOOTSTRAP_SERVERS"]
# S3_BUCKET = config["S3_BUCKET"]

# # Topic injected per container via environment variable
# TOPIC = os.environ["KAFKA_TOPIC"]

# s3 = boto3.client("s3", region_name="eu-north-1")


# # --- S3 Write ---
# def write_to_s3(topic: str, payload: dict):
#     now = datetime.now(timezone.utc)
#     is_dlq = topic.endswith("-dlq")
#     prefix = "errors/producer" if is_dlq else f"raw/{topic}"
#     key = f"{prefix}/{now.year}/{now.month:02d}/{now.day:02d}/{now.timestamp()}.json"

#     s3.put_object(
#         Bucket=S3_BUCKET,
#         Key=key,
#         Body=json.dumps(payload),
#         ContentType="application/json"
#     )
#     logger.info(f"[{topic}] Written to s3://{S3_BUCKET}/{key}")


# # --- Consumer ---
# def run_consumer():
#     logger.info(f"[{TOPIC}] Starting consumer")
#     consumer = KafkaConsumer(
#         TOPIC,
#         bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
#         value_deserializer=lambda v: json.loads(v.decode("utf-8")),
#         auto_offset_reset="earliest",
#         enable_auto_commit=False,
#         group_id=f"consumer-{TOPIC}",
#         session_timeout_ms=30000,
#         heartbeat_interval_ms=10000,
#     )

#     for message in consumer:
#         try:
#             write_to_s3(TOPIC, message.value)
#             consumer.commit()
#         except Exception as e:
#             logger.error(f"[{TOPIC}] Failed to write to S3: {e} — offset not committed, will retry")


# if __name__ == "__main__":
#     run_consumer()




import json
import sys
import boto3
import logging
import os
import time
from kafka import KafkaConsumer
from pydantic import ValidationError
from datetime import datetime, timezone

sys.path.insert(0, "/app")
from contracts.trade import TradeEvent
from contracts.orderbook import OrderbookEvent

CONTRACTS = {
    "crypto-prices": TradeEvent,
    "crypto-orderbook": OrderbookEvent,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


config = get_secret(os.environ["SECRET_CONSUMER"])
KAFKA_BOOTSTRAP_SERVERS = config["KAFKA_BOOTSTRAP_SERVERS"]
S3_BUCKET = config["S3_BUCKET"]

TOPIC = os.environ["KAFKA_TOPIC"]
BATCH_SIZE = 1000
BATCH_INTERVAL = 60

s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])


def write_batch_to_s3(topic: str, batch: list):
    now = datetime.now(timezone.utc)
    is_dlq = topic.endswith("-dlq")
    prefix = "errors/producer" if is_dlq else f"raw/{topic}"
    key = f"{prefix}/{now.year}/{now.month:02d}/{now.day:02d}/{int(now.timestamp())}.json"
    body = "\n".join(json.dumps(record) for record in batch)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json"
    )
    logger.info(f"[{topic}] Written batch of {len(batch)} records to s3://{S3_BUCKET}/{key}")


def run_consumer():
    logger.info(f"[{TOPIC}] Starting consumer — batch size: {BATCH_SIZE}, interval: {BATCH_INTERVAL}s")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=f"consumer-{TOPIC}",
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000,
    )

    batch = []
    last_write_time = time.time()

    contract = CONTRACTS.get(TOPIC)

    for message in consumer:
        record = message.value
        if contract:
            try:
                contract(**record)
            except ValidationError as e:
                logger.error(f"[CONTRACT VIOLATION] {TOPIC}: {e} — skipping record")
                continue
        batch.append(record)
        time_elapsed = time.time() - last_write_time
        should_write = len(batch) >= BATCH_SIZE or time_elapsed >= BATCH_INTERVAL

        if should_write and batch:
            try:
                write_batch_to_s3(TOPIC, batch)
                consumer.commit()
                batch = []
                last_write_time = time.time()
            except Exception as e:
                logger.error(f"[{TOPIC}] Failed to write batch: {e} — will retry")


if __name__ == "__main__":
    run_consumer()