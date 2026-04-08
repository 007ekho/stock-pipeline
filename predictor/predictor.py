import json
import boto3
import logging
import numpy as np
from io import BytesIO
from collections import deque
from kafka import KafkaConsumer, KafkaProducer
import joblib
import tensorflow as tf
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SEQUENCE_LENGTH = 60
FEATURE_COLS = [
    "price", "size",
    "price_lag_1", "price_lag_5", "price_lag_10",
    "price_change", "price_change_pct",
    "rolling_avg_5", "rolling_avg_10",
    "rolling_stddev_10", "volume_change",
    "spread", "bid_ask_ratio",
    "total_bid_volume", "total_ask_volume",
]
BUFFER_SIZE = SEQUENCE_LENGTH + 10  # extra room for lag computation


def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name="eu-north-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


config = get_secret("stock-pipeline/producer")
KAFKA_BOOTSTRAP_SERVERS = config["KAFKA_BOOTSTRAP_SERVERS"]

s3 = boto3.client("s3", region_name="eu-north-1")
s3_config = get_secret("stock-pipeline/spark")
S3_BUCKET = s3_config["S3_BUCKET"]


def load_latest_model(symbol: str):
    """Load the latest model and scalers for a symbol from S3."""
    prefix = f"processed/models/{symbol}/"
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    objects = response.get("Contents", [])
    if not objects:
        logger.warning(f"No model found for {symbol}")
        return None, None, None

    # Find latest timestamp folder
    folders = set(obj["Key"].split("/")[3] for obj in objects)
    latest = sorted(folders)[-1]
    base = f"{prefix}{latest}"

    # Load model
    model_obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{base}/model.keras")
    model_path = f"/tmp/{symbol}_model.keras"
    with open(model_path, "wb") as f:
        f.write(model_obj["Body"].read())
    model = tf.keras.models.load_model(model_path)

    # Load scalers
    def load_scaler(name):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{base}/{name}.joblib")
        return joblib.load(BytesIO(obj["Body"].read()))

    feature_scaler = load_scaler("feature_scaler")
    price_scaler = load_scaler("price_scaler")

    logger.info(f"Loaded model + scalers for {symbol} from {base}")
    return model, feature_scaler, price_scaler


def compute_features(buffer: list) -> dict | None:
    """Compute features for the latest record using the rolling buffer."""
    if len(buffer) < 11:  # need at least 11 records for lag_10
        return None

    latest = buffer[-1]
    prices = [r["price"] for r in buffer]
    sizes = [r["size"] for r in buffer]

    lag1 = prices[-2]
    lag5 = prices[-6] if len(prices) >= 6 else prices[0]
    lag10 = prices[-11] if len(prices) >= 11 else prices[0]

    price_change = latest["price"] - lag1
    price_change_pct = (price_change / lag1 * 100) if lag1 != 0 else 0.0

    recent5 = prices[-5:]
    recent10 = prices[-10:]
    rolling_avg_5 = float(np.mean(recent5))
    rolling_avg_10 = float(np.mean(recent10))
    rolling_stddev_10 = float(np.std(recent10)) if len(recent10) > 1 else 0.0
    volume_change = sizes[-1] - sizes[-2] if len(sizes) >= 2 else 0.0

    return {
        "price": latest["price"],
        "size": latest["size"],
        "price_lag_1": lag1,
        "price_lag_5": lag5,
        "price_lag_10": lag10,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "rolling_avg_5": rolling_avg_5,
        "rolling_avg_10": rolling_avg_10,
        "rolling_stddev_10": rolling_stddev_10,
        "volume_change": volume_change,
        "spread": latest.get("spread", 0.0),
        "bid_ask_ratio": latest.get("bid_ask_ratio", 0.0),
        "total_bid_volume": latest.get("total_bid_volume", 0.0),
        "total_ask_volume": latest.get("total_ask_volume", 0.0),
    }


class Predictor:
    def __init__(self):
        # Rolling buffers of raw trade + orderbook records per symbol
        self.buffers: dict[str, deque] = {s: deque(maxlen=BUFFER_SIZE) for s in SYMBOLS}
        # Latest orderbook snapshot per symbol
        self.orderbook: dict[str, dict] = {s: {} for s in SYMBOLS}
        # Feature buffers (computed features, ready for model input)
        self.feature_buffers: dict[str, deque] = {s: deque(maxlen=SEQUENCE_LENGTH) for s in SYMBOLS}
        # Models and scalers
        self.models = {}
        self.feature_scalers = {}
        self.price_scalers = {}

        self._load_all_models()

    def _load_all_models(self):
        for symbol in SYMBOLS:
            model, feature_scaler, price_scaler = load_latest_model(symbol)
            if model:
                self.models[symbol] = model
                self.feature_scalers[symbol] = feature_scaler
                self.price_scalers[symbol] = price_scaler

    def on_price(self, record: dict):
        symbol = record.get("symbol")
        if symbol not in SYMBOLS:
            return

        # Merge with latest orderbook
        ob = self.orderbook.get(symbol, {})
        entry = {
            "price": record["price"],
            "size": record["size"],
            "timestamp": record["timestamp"],
            "spread": ob.get("top_ask_price", 0) - ob.get("top_bid_price", 0),
            "bid_ask_ratio": ob.get("bid_ask_ratio", 0.0),
            "total_bid_volume": ob.get("total_bid_volume", 0.0),
            "total_ask_volume": ob.get("total_ask_volume", 0.0),
        }
        self.buffers[symbol].append(entry)

        features = compute_features(list(self.buffers[symbol]))
        if features is None:
            return

        self.feature_buffers[symbol].append([features[c] for c in FEATURE_COLS])

        if len(self.feature_buffers[symbol]) == SEQUENCE_LENGTH:
            self._predict(symbol, record["price"])

    def on_orderbook(self, record: dict):
        symbol = record.get("symbol")
        if symbol in SYMBOLS:
            self.orderbook[symbol] = record

    def _predict(self, symbol: str, current_price: float):
        if symbol not in self.models:
            return

        model = self.models[symbol]
        feature_scaler = self.feature_scalers[symbol]
        price_scaler = self.price_scalers[symbol]

        sequence = np.array(list(self.feature_buffers[symbol]))  # (60, 15)
        try:
            scaled = feature_scaler.transform(sequence)
            X = scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLS))
            pred_scaled = model.predict(X, verbose=0)
            predicted_price = float(price_scaler.inverse_transform(pred_scaled)[0][0])
            direction = "UP" if predicted_price > current_price else "DOWN"
            diff = predicted_price - current_price
            logger.info(
                f"[{symbol}] current={current_price:.4f} | "
                f"predicted={predicted_price:.4f} | "
                f"diff={diff:+.4f} | {direction}"
            )
        except Exception as e:
            logger.error(f"[{symbol}] Prediction error: {e}")


def run():
    predictor = Predictor()

    if not predictor.models:
        logger.error("No models loaded — train models first then restart predictor")
        return

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    # Patch predictor to also publish to Kafka
    original_predict = predictor._predict

    def predict_and_publish(symbol: str, current_price: float):
        if symbol not in predictor.models:
            return
        model = predictor.models[symbol]
        feature_scaler = predictor.feature_scalers[symbol]
        price_scaler = predictor.price_scalers[symbol]
        sequence = np.array(list(predictor.feature_buffers[symbol]))
        try:
            scaled = feature_scaler.transform(sequence)
            X = scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLS))
            pred_scaled = model.predict(X, verbose=0)
            predicted_price = float(price_scaler.inverse_transform(pred_scaled)[0][0])
            direction = "UP" if predicted_price > current_price else "DOWN"
            diff = predicted_price - current_price
            logger.info(
                f"[{symbol}] current={current_price:.4f} | "
                f"predicted={predicted_price:.4f} | "
                f"diff={diff:+.4f} | {direction}"
            )
            producer.send("crypto-predictions", value={
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "direction": direction,
                "diff": round(diff, 4),
                "timestamp": int(__import__("time").time() * 1000),
            })
        except Exception as e:
            logger.error(f"[{symbol}] Prediction error: {e}")

    predictor._predict = predict_and_publish

    prices_consumer = KafkaConsumer(
        "crypto-prices",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="predictor-prices",
    )
    orderbook_consumer = KafkaConsumer(
        "crypto-orderbook",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="predictor-orderbook",
    )

    def consume_orderbook():
        for msg in orderbook_consumer:
            predictor.on_orderbook(msg.value)

    threading.Thread(target=consume_orderbook, daemon=True).start()
    logger.info("Predictor running — waiting for trades...")

    for msg in prices_consumer:
        predictor.on_price(msg.value)


if __name__ == "__main__":
    run()
