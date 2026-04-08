import websocket
from datetime import datetime, timezone
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
KAFKA_TOPIC = "crypto-orderbook"
DLQ_TOPIC = "crypto-orderbook-dlq"

# --- Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    retries=5,
    retry_backoff_ms=500,
    request_timeout_ms=30000,
    acks="all"
)

STREAM_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@1000ms/ethusdt@depth20@1000ms/solusdt@depth20@1000ms"


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
    logger.info("Connected to Binance Order Book WebSocket")


def on_message(ws, message):
    try:
        data = json.loads(message)
        stream = data.get("stream", "")
        event = data.get("data", {})

        symbol = stream.split("@")[0].upper()
        bids = event.get("bids", [])
        asks = event.get("asks", [])

        if not bids or not asks:
            logger.warning(f"Empty order book for {symbol} — skipping")
            return

        payload = {
            "symbol": symbol,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "top_bid_price": float(bids[0][0]),
            "top_bid_qty": float(bids[0][1]),
            "top_ask_price": float(asks[0][0]),
            "top_ask_qty": float(asks[0][1]),
            "bid_depth": [[float(p), float(q)] for p, q in bids],
            "ask_depth": [[float(p), float(q)] for p, q in asks],
            "total_bid_volume": sum(float(q) for _, q in bids),
            "total_ask_volume": sum(float(q) for _, q in asks),
            "bid_ask_ratio": sum(float(q) for _, q in bids) / sum(float(q) for _, q in asks),
        }

        logger.info(f"Orderbook: {symbol} | top bid: {payload['top_bid_price']} | top ask: {payload['top_ask_price']} | ratio: {payload['bid_ask_ratio']:.3f}")
        send_with_retry(KAFKA_TOPIC, payload)

    except Exception as e:
        logger.error(f"Failed to process message: {e} — skipping")


def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")


# --- Run ---
if __name__ == "__main__":
    logger.info("Starting order book producer...")
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
            logger.error(f"Producer2 crashed: {e}. Restarting in 5s...")
            time.sleep(5)