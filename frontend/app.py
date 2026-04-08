import streamlit as st
import pandas as pd
from collections import defaultdict, deque
from kafka import KafkaConsumer
import json
import threading
import time
import os

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MAX_POINTS = 200

# --- Shared state (module-level, persists across reruns) ---
prices_data = defaultdict(lambda: deque(maxlen=MAX_POINTS))       # symbol -> deque of {t, price}
predictions_data = defaultdict(lambda: deque(maxlen=MAX_POINTS))  # symbol -> deque of {t, predicted_price, direction}
_consumers_started = False


def start_consumers():
    global _consumers_started
    if _consumers_started:
        return
    _consumers_started = True

    def consume_prices():
        consumer = KafkaConsumer(
            "crypto-prices",
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="frontend-prices",
        )
        for msg in consumer:
            d = msg.value
            prices_data[d["symbol"]].append({
                "t": d["timestamp"],
                "price": d["price"],
            })

    def consume_predictions():
        consumer = KafkaConsumer(
            "crypto-predictions",
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="frontend-predictions",
        )
        for msg in consumer:
            d = msg.value
            predictions_data[d["symbol"]].append({
                "t": d["timestamp"],
                "predicted_price": d["predicted_price"],
                "direction": d["direction"],
                "diff": d["diff"],
                "current_price": d["current_price"],
            })

    threading.Thread(target=consume_prices, daemon=True).start()
    threading.Thread(target=consume_predictions, daemon=True).start()


def build_chart_df(symbol: str) -> pd.DataFrame:
    prices = list(prices_data[symbol])
    preds = list(predictions_data[symbol])

    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(prices).rename(columns={"price": "Actual Price"}).set_index("t")
    if preds:
        pred_df = pd.DataFrame(preds)[["t", "predicted_price"]].rename(
            columns={"predicted_price": "Predicted Price"}
        ).set_index("t")
        return price_df.join(pred_df, how="outer").sort_index().tail(MAX_POINTS)
    return price_df.sort_index().tail(MAX_POINTS)


# --- App ---
st.set_page_config(page_title="Crypto Live Predictions", layout="wide")
st.title("Crypto Price Predictions")

start_consumers()

placeholder = st.empty()

while True:
    with placeholder.container():
        any_data = any(prices_data[s] for s in SYMBOLS)

        if not any_data:
            st.info("Waiting for live data from Kafka...")
        else:
            for symbol in SYMBOLS:
                if not prices_data[symbol]:
                    continue

                preds = list(predictions_data[symbol])
                latest_price = list(prices_data[symbol])[-1]["price"]
                latest_pred = preds[-1] if preds else None

                st.subheader(symbol)

                # Metrics row
                col1, col2, col3 = st.columns(3)
                col1.metric("Live Price", f"${latest_price:,.4f}")
                if latest_pred:
                    col2.metric(
                        "Predicted Next",
                        f"${latest_pred['predicted_price']:,.4f}",
                        delta=f"{latest_pred['diff']:+.4f}",
                    )
                    col3.metric("Direction", latest_pred["direction"])

                # Chart
                df = build_chart_df(symbol)
                if not df.empty:
                    st.line_chart(df, height=250)

                # Recent predictions table
                if preds:
                    recent = pd.DataFrame(preds[-10:][::-1])[
                        ["current_price", "predicted_price", "diff", "direction"]
                    ].rename(columns={
                        "current_price": "Actual",
                        "predicted_price": "Predicted",
                        "diff": "Diff",
                        "direction": "Direction",
                    })
                    st.dataframe(recent, use_container_width=True, hide_index=True)

                st.divider()

    time.sleep(1)
