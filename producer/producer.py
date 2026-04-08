# import websocket
# import json
# import boto3
# from kafka import KafkaProducer
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # --- Secrets ---
# def get_secret(secret_name: str) -> dict:
#     import os
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
#     response = client.get_secret_value(SecretId=secret_name)
#     return json.loads(response["SecretString"])


# config = get_secret(os.environ["SECRET_PRODUCER"])
# KAFKA_BOOTSTRAP_SERVERS = config["KAFKA_BOOTSTRAP_SERVERS"]
# KAFKA_TOPIC = "crypto-prices"

# # --- Kafka Producer ---
# producer = KafkaProducer(
#     bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
#     value_serializer=lambda v: json.dumps(v).encode("utf-8")
# )

# # Binance stream URL — multiple symbols combined into one stream
# STREAM_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/solusdt@trade"


# # --- Handlers ---
# def on_open(ws):
#     logger.info("Connected to Binance WebSocket")


# def on_message(ws, message):
#     data = json.loads(message)
#     event = data.get("data", {})

#     if event.get("e") == "trade":
#         payload = {
#             "symbol": event.get("s"),       # e.g. BTCUSDT
#             "price": float(event.get("p")), # trade price
#             "size": float(event.get("q")),  # trade quantity
#             "timestamp": event.get("T"),    # trade time (ms)
#             "trade_id": event.get("t"),
#         }
#         logger.info(f"Trade: {payload}")
#         producer.send(KAFKA_TOPIC, value=payload)


# def on_error(ws, error):
#     logger.error(f"WebSocket error: {error}")


# def on_close(ws, close_status_code, close_msg):
#     logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")


# # --- Run ---
# if __name__ == "__main__":
#     logger.info("Starting producer...")
#     ws = websocket.WebSocketApp(
#         STREAM_URL,
#         on_open=on_open,
#         on_message=on_message,
#         on_error=on_error,
#         on_close=on_close,
#     )
#     ws.run_forever()


import websocket
import json
import os
import boto3
from kafka import KafkaProducer
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Secrets ---
def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


config = get_secret(os.environ["SECRET_PRODUCER"])
KAFKA_BOOTSTRAP_SERVERS = config["KAFKA_BOOTSTRAP_SERVERS"]
KAFKA_TOPIC = "crypto-prices"
DLQ_TOPIC = "crypto-prices-dlq"

# --- Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    retries=5,
    retry_backoff_ms=500,
    request_timeout_ms=30000,
    acks="all"
)

STREAM_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/solusdt@trade"


# --- Kafka Send with DLQ ---
def send_with_retry(topic: str, payload: dict):
    try:
        future = producer.send(topic, value=payload)
        future.get(timeout=10)
    except Exception as e:
        logger.error(f"Failed to send to {topic}: {e} — sending to DLQ")
        try:
            producer.send(DLQ_TOPIC, value={
                "original_payload": payload,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as dlq_error:
            logger.critical(f"DLQ also failed: {dlq_error}")


# --- Handlers ---
def on_open(ws):
    logger.info("Connected to Binance WebSocket")


def on_message(ws, message):
    try:
        data = json.loads(message)
        event = data.get("data", {})

        if event.get("e") == "trade":
            payload = {
                "symbol": event.get("s"),
                "price": float(event.get("p")),
                "size": float(event.get("q")),
                "timestamp": event.get("T"),
                "trade_id": event.get("t"),
            }
            logger.info(f"Trade: {payload}")
            send_with_retry(KAFKA_TOPIC, payload)

    except Exception as e:
        logger.error(f"Failed to process message: {e} — skipping")


def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")


# --- Run ---
if __name__ == "__main__":
    logger.info("Starting producer...")
    while True:
        try:
            ws = websocket.WebSocketApp(
                STREAM_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(
                ping_interval=30,
                ping_timeout=10,
                reconnect=5
            )
        except Exception as e:
            logger.error(f"Producer crashed: {e}. Restarting in 5s...")
            time.sleep(5)
