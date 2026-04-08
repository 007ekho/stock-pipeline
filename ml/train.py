# import boto3
# import json
# import numpy as np
# import pandas as pd
# import logging
# from io import BytesIO
# from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # --- Secrets ---
# def get_secret(secret_name: str) -> dict:
#     client = boto3.client("secretsmanager", region_name="eu-north-1")
#     response = client.get_secret_value(SecretId=secret_name)
#     return json.loads(response["SecretString"])


# config = get_secret("stock-pipeline/ml")
# S3_BUCKET = config["S3_BUCKET"]

# s3 = boto3.client("s3", region_name="eu-north-1")

# SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
# SEQUENCE_LENGTH = 60       # use last 60 trades to predict next price
# PREDICTION_HORIZON = 1     # predict 1 step ahead
# TRAIN_SPLIT = 0.8
# EPOCHS = 2
# BATCH_SIZE = 32

# FEATURE_COLS = [
#     "price", "size",
#     "price_lag_1", "price_lag_5", "price_lag_10",
#     "price_change", "price_change_pct",
#     "rolling_avg_5", "rolling_avg_10",
#     "rolling_stddev_10", "volume_change",
#     "spread", "bid_ask_ratio",
#     "total_bid_volume", "total_ask_volume",
# ]


# # --- Load training data from S3 ---
# def load_training_data() -> pd.DataFrame:
#     logger.info("Loading training data from S3...")
#     paginator = s3.get_paginator("list_objects_v2")
#     pages = paginator.paginate(Bucket=S3_BUCKET, Prefix="processed/training/")

#     dfs = []
#     for page in pages:
#         for obj in page.get("Contents", []):
#             if obj["Key"].endswith(".parquet"):
#                 response = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
#                 df = pd.read_parquet(BytesIO(response["Body"].read()))
#                 dfs.append(df)

#     if not dfs:
#         raise ValueError("No training data found in S3")

#     df = pd.concat(dfs, ignore_index=True)
#     logger.info(f"Loaded {len(df)} records from S3")
#     return df


# # --- Prepare sequences for LSTM ---
# def create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i + seq_length])
#         y.append(target[i + seq_length])
#     return np.array(X), np.array(y)


# # --- Build LSTM model ---
# def build_model(input_shape: tuple) -> tf.keras.Model:
#     model = Sequential([
#         LSTM(128, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(64, return_sequences=False),
#         Dropout(0.2),
#         Dense(32, activation="relu"),
#         Dense(1)  # predict next price
#     ])
#     model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#     return model


# # --- Save model to S3 ---
# def save_model_to_s3(model: tf.keras.Model, symbol: str, metrics: dict):
#     now = datetime.utcnow()
#     key = f"processed/models/{symbol}/{now.strftime('%Y%m%d_%H%M%S')}/model.keras"

#     # Save locally then upload
#     local_path = f"/tmp/{symbol}_model.keras"
#     model.save(local_path)

#     s3.upload_file(local_path, S3_BUCKET, key)
#     logger.info(f"Saved model to s3://{S3_BUCKET}/{key}")

#     # Save metrics
#     metrics_key = f"processed/models/{symbol}/{now.strftime('%Y%m%d_%H%M%S')}/metrics.json"
#     s3.put_object(
#         Bucket=S3_BUCKET,
#         Key=metrics_key,
#         Body=json.dumps(metrics),
#         ContentType="application/json"
#     )
#     logger.info(f"Saved metrics to s3://{S3_BUCKET}/{metrics_key}")


# # --- Train per symbol ---
# def train_symbol(df: pd.DataFrame, symbol: str):
#     logger.info(f"Training LSTM for {symbol}...")

#     # Filter to symbol and sort by timestamp
#     df_symbol = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)

#     if len(df_symbol) < SEQUENCE_LENGTH + 10:
#         logger.warning(f"Not enough data for {symbol} — skipping")
#         return

#     logger.info(f"{symbol}: {len(df_symbol)} records")

#     # Scale features
#     feature_scaler = MinMaxScaler()
#     price_scaler = MinMaxScaler()

#     features = feature_scaler.fit_transform(df_symbol[FEATURE_COLS].values)
#     prices = price_scaler.fit_transform(df_symbol[["price"]].values)

#     # Create sequences
#     X, y = create_sequences(features, prices.flatten(), SEQUENCE_LENGTH)

#     # Train/test split
#     split = int(len(X) * TRAIN_SPLIT)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     logger.info(f"{symbol}: {len(X_train)} train, {len(X_test)} test sequences")

#     # Build and train model
#     model = build_model(input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLS)))
#     model.summary()

#     early_stop = EarlyStopping(
#         monitor="val_loss",
#         patience=5,
#         restore_best_weights=True
#     )

#     history = model.fit(
#         X_train, y_train,
#         validation_split=0.1,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         callbacks=[early_stop],
#         verbose=1
#     )

#     # Evaluate
#     y_pred = model.predict(X_test)
#     y_pred_actual = price_scaler.inverse_transform(y_pred)
#     y_test_actual = price_scaler.inverse_transform(y_test.reshape(-1, 1))

#     mae = mean_absolute_error(y_test_actual, y_pred_actual)
#     rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
#     mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

#     metrics = {
#         "symbol": symbol,
#         "mae": float(mae),
#         "rmse": float(rmse),
#         "mape": float(mape),
#         "train_records": len(X_train),
#         "test_records": len(X_test),
#         "epochs_trained": len(history.history["loss"]),
#         "trained_at": datetime.utcnow().isoformat()
#     }

#     logger.info(f"{symbol} — MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

#     # Save model and metrics to S3
#     save_model_to_s3(model, symbol, metrics)
#     return metrics


# # --- Main ---
# if __name__ == "__main__":
#     logger.info("Starting LSTM training pipeline...")

#     df = load_training_data()

#     all_metrics = []
#     for symbol in SYMBOLS:
#         metrics = train_symbol(df, symbol)
#         if metrics:
#             all_metrics.append(metrics)

#     logger.info("Training complete. Summary:")
#     for m in all_metrics:
#         logger.info(
#             f"{m['symbol']} — MAE: {m['mae']:.4f} | "
#             f"RMSE: {m['rmse']:.4f} | "
#             f"MAPE: {m['mape']:.2f}%"
#         )


import boto3
import json
import numpy as np
import pandas as pd
import logging
from io import BytesIO
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name="eu-north-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


config = get_secret("stock-pipeline/ml")
S3_BUCKET = config["S3_BUCKET"]
s3 = boto3.client("s3", region_name="eu-north-1")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SEQUENCE_LENGTH = 60
TRAIN_SPLIT = 0.8
EPOCHS = 2
BATCH_SIZE = 32

FEATURE_COLS = [
    "price", "size",
    "price_lag_1", "price_lag_5", "price_lag_10",
    "price_change", "price_change_pct",
    "rolling_avg_5", "rolling_avg_10",
    "rolling_stddev_10", "volume_change",
    "spread", "bid_ask_ratio",
    "total_bid_volume", "total_ask_volume",
]


def load_training_data() -> pd.DataFrame:
    logger.info("Loading training data from S3...")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix="processed/training/")

    dfs = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                response = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                df = pd.read_parquet(BytesIO(response["Body"].read()))
                dfs.append(df)

    if not dfs:
        raise ValueError("No training data found in S3")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} records from S3")
    return df


def clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean data — remove NaN and inf values."""
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill orderbook nulls with 0
    fill_cols = ["bid_ask_ratio", "total_bid_volume", "total_ask_volume", "spread"]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Fill remaining nulls with forward fill then 0
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().fillna(0)

    # Drop any remaining NaN
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    after = len(df)

    logger.info(f"{symbol}: dropped {before - after} rows with NaN")
    logger.info(f"{symbol}: NaN remaining: {df[FEATURE_COLS].isna().sum().sum()}")
    logger.info(f"{symbol}: Inf remaining: {np.isinf(df[FEATURE_COLS].values).sum()}")

    return df


def create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def build_model(input_shape: tuple) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def save_model_to_s3(model: tf.keras.Model, symbol: str, metrics: dict, feature_scaler, price_scaler):
    import joblib
    from io import BytesIO

    now = datetime.utcnow()
    prefix = f"processed/models/{symbol}/{now.strftime('%Y%m%d_%H%M%S')}"

    # Save model
    local_path = f"/tmp/{symbol}_model.keras"
    model.save(local_path)
    s3.upload_file(local_path, S3_BUCKET, f"{prefix}/model.keras")
    logger.info(f"Saved model to s3://{S3_BUCKET}/{prefix}/model.keras")

    # Save scalers
    for name, scaler in [("feature_scaler", feature_scaler), ("price_scaler", price_scaler)]:
        buf = BytesIO()
        joblib.dump(scaler, buf)
        buf.seek(0)
        s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/{name}.joblib", Body=buf.read())
        logger.info(f"Saved {name} to s3://{S3_BUCKET}/{prefix}/{name}.joblib")

    # Save metrics
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{prefix}/metrics.json",
        Body=json.dumps(metrics),
        ContentType="application/json"
    )
    logger.info(f"Saved metrics to s3://{S3_BUCKET}/{prefix}/metrics.json")


def train_symbol(df: pd.DataFrame, symbol: str):
    logger.info(f"Training LSTM for {symbol}...")

    # Filter and sort
    df_symbol = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)

    if len(df_symbol) < SEQUENCE_LENGTH + 10:
        logger.warning(f"Not enough data for {symbol} — skipping")
        return

    # Clean
    df_symbol = clean_data(df_symbol, symbol)

    if len(df_symbol) < SEQUENCE_LENGTH + 10:
        logger.warning(f"Not enough data for {symbol} after cleaning — skipping")
        return

    logger.info(f"{symbol}: {len(df_symbol)} records after cleaning")

    # Scale
    feature_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()

    features = feature_scaler.fit_transform(df_symbol[FEATURE_COLS].values)
    prices = price_scaler.fit_transform(df_symbol[["price"]].values)

    # Final NaN check after scaling
    if np.isnan(features).any():
        logger.error(f"{symbol}: NaN after scaling — skipping")
        return

    # Sequences
    X, y = create_sequences(features, prices.flatten(), SEQUENCE_LENGTH)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info(f"{symbol}: {len(X_train)} train, {len(X_test)} test sequences")

    # Train
    model = build_model(input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLS)))
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_actual = price_scaler.inverse_transform(y_pred)
    y_test_actual = price_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Check for NaN in predictions
    if np.isnan(y_pred_actual).any():
        logger.error(f"{symbol}: NaN in predictions — model failed to learn")
        return

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    metrics = {
        "symbol": symbol,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "epochs_trained": len(history.history["loss"]),
        "trained_at": datetime.utcnow().isoformat()
    }

    logger.info(f"{symbol} — MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
    save_model_to_s3(model, symbol, metrics, feature_scaler, price_scaler)
    return metrics


if __name__ == "__main__":
    logger.info("Starting LSTM training pipeline...")
    df = load_training_data()

    all_metrics = []
    for symbol in SYMBOLS:
        metrics = train_symbol(df, symbol)
        if metrics:
            all_metrics.append(metrics)

    logger.info("Training complete. Summary:")
    for m in all_metrics:
        logger.info(
            f"{m['symbol']} — MAE: {m['mae']:.4f} | "
            f"RMSE: {m['rmse']:.4f} | "
            f"MAPE: {m['mape']:.2f}%"
        )