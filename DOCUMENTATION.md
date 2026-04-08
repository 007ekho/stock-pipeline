# Stock Pipeline — Project Documentation

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Lessons Learned](#2-lessons-learned)
3. [Docker Command Reference](#3-docker-command-reference)
4. [Airflow Operations](#4-airflow-operations)
5. [AWS & S3 Reference](#5-aws--s3-reference)
6. [Environment Variables (.env)](#6-environment-variables-env)
7. [Data Stream Reference](#7-data-stream-reference)
8. [Data Contracts](#8-data-contracts)
9. [Production Way Forward](#9-production-way-forward)

---

## 1. Project Summary

A real-time cryptocurrency data pipeline built entirely on Docker, streaming live trade and orderbook data from Binance into a full ML training and inference loop.

### What It Does
- Streams live BTC, ETH, SOL trades and orderbook snapshots via WebSocket
- Persists raw data to AWS S3 via Kafka consumers
- Runs nightly Spark batch jobs to merge, join, and engineer 15 trading features
- Trains an LSTM model per symbol and saves model + scalers to S3
- Runs a real-time predictor that loads the latest model and publishes next-trade price predictions
- Displays live and predicted prices on a Streamlit dashboard

### Stack
| Layer | Technology |
|---|---|
| Data source | Binance WebSocket API (free, no key required) |
| Message broker | Apache Kafka 7.7.0 (KRaft, single broker) |
| Stream persistence | Python kafka-python consumers → AWS S3 |
| Batch processing | Apache Spark 3.x (standalone cluster, 2 workers, PySpark) |
| Orchestration | Apache Airflow (SequentialExecutor + SQLite) |
| ML training | TensorFlow/Keras LSTM, scikit-learn MinMaxScaler |
| Inference | Python predictor with rolling buffer |
| Frontend | Streamlit |
| Data contracts | Pydantic (Kafka boundary) + schema checks (Spark layers) |
| Infrastructure | Docker Compose (16 containers) |
| Cloud | AWS S3 (eu-north-1), AWS Secrets Manager |

---

## 2. Lessons Learned

### 2.1 S3AFileSystem — Bake JARs Into the Image, Don't Resolve at Runtime

**Problem:** Spark jobs failed with `ClassNotFoundException: org.apache.hadoop.fs.s3a.S3AFileSystem` when using `spark.jars.packages` to download hadoop-aws JARs from Maven at runtime inside Docker.

**Root cause:** Maven resolution inside Docker often fails due to network restrictions, missing mirrors, or timeout. The `spark.jars.packages` config tries to download JARs every time the job starts — unreliable in a containerised environment.

**Fix:** Pre-bake the JARs directly into the Spark and Airflow Dockerfiles:
```dockerfile
RUN curl -fL -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -fL -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
```

**Key insight:** The Spark *driver* runs inside the Airflow container (client mode), not the Spark container. Both images need the JARs — not just the Spark image.

**Remove this from all Spark job scripts:**
```python
# REMOVE THIS — causes runtime Maven resolution failure
.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,...")
```

---

### 2.2 SQLite Locking Kills Long-Running Airflow Tasks

**Problem:** Long Spark or ML training tasks caused `sqlalchemy.exc.OperationalError: database is locked` in Airflow, marking tasks as `up_for_retry` or `failed` mid-execution.

**Root cause:** Airflow's SequentialExecutor uses SQLite as its metadata database. SQLite has a single writer lock — long-running tasks hold open transactions that block the scheduler's heartbeat writes, eventually timing out.

**Fix (short-term):** Reduce training data volume. We added `--limit-per-symbol 10000` to `prepare_training.py` using a Window function to take the most recent N records per symbol:
```python
w = Window.partitionBy("symbol").orderBy(desc("timestamp"))
df = df.withColumn("_row", row_number().over(w)) \
       .filter(col("_row") <= args.limit_per_symbol) \
       .drop("_row")
```
This reduced training from 5,152 steps/epoch to 224 steps/epoch — cutting task duration from 40+ minutes to ~3 minutes.

**Fix (production):** Switch Airflow to PostgreSQL + CeleryExecutor. SQLite is explicitly not supported for production by the Airflow project.

---

### 2.3 DockerOperator Requires Docker Socket + Host Path Resolution

**Problem:** The Airflow `DockerOperator` for ML training needs to spawn a container on the host Docker daemon, not inside the Airflow container.

**Required setup in docker-compose.yaml:**
```yaml
x-airflow-common:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock  # grants Docker access
  environment:
    - HOST_HOME=${HOME}  # resolves host ~ path for AWS credential mount
```

**Required in the DAG:**
```python
from docker.types import Mount
import os

DockerOperator(
    image="stock-pipeline-ml",
    mounts=[Mount(
        source=f"{os.environ['HOST_HOME']}/.aws",
        target="/root/.aws",
        type="bind",
        read_only=True,
    )],
    docker_url="unix://var/run/docker.sock",
    auto_remove="success",
)
```

**Key insight:** `~/.aws` inside the Airflow container resolves to the Airflow container's home, not the host's. `HOST_HOME` is passed in as an env var so the mount path resolves correctly on the host daemon.

**Also required:** The ML image must be pre-built and tagged before Airflow triggers it:
```bash
docker-compose build ml
# image: stock-pipeline-ml  # must be set in docker-compose.yaml ml service
```

---

### 2.4 Scaler Persistence Is Required for Correct Inference

**Problem:** Initial predictor produced nonsensical predictions because the model's output was still in scaled space (0–1) and couldn't be inverse-transformed — the scalers were never saved.

**Fix:** Save both `feature_scaler` and `price_scaler` as joblib files alongside the model in S3:
```python
import joblib
from io import BytesIO

for name, scaler in [("feature_scaler", feature_scaler), ("price_scaler", price_scaler)]:
    buf = BytesIO()
    joblib.dump(scaler, buf)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/{name}.joblib", Body=buf.read())
```

**Rule:** Training and inference must use *identical* scalers. The scaler fitted on training data must be loaded at inference time — never refit on live data.

---

### 2.5 Feature Parity Between Training and Inference Is Critical

**Problem:** If the predictor computes features differently from `feature_engineering.py`, the model receives out-of-distribution inputs and predictions degrade silently.

**Fix:** The predictor's rolling buffer replicates all 15 features exactly as computed in Spark:
```
price, size,
price_lag_1, price_lag_5, price_lag_10,
price_change, price_change_pct,
rolling_avg_5, rolling_avg_10,
rolling_stddev_10, volume_change,
spread, bid_ask_ratio,
total_bid_volume, total_ask_volume
```

**Rule:** Any change to `feature_engineering.py` must be mirrored in `predictor.py`, and the model must be retrained before the new features are used in inference.

---

### 2.6 DAG Collision — `daily_processing` and `ml_retraining` Competing for the Same Spark Worker

**Problem:** Both DAGs were scheduled at midnight (`0 0 * * *`). On days when both were due, they submitted Spark jobs simultaneously to a single worker. With SequentialExecutor, the second DAG's tasks queued behind the first, and retries from failed runs piled on top — the worker was never free, and the scheduler's SQLite heartbeat timed out under the load.

**Root cause:** Three compounding issues:
1. Single Spark worker — no capacity for concurrent jobs
2. Identical midnight schedules — both DAGs fired at the same moment
3. No `max_active_runs` guard — retries stacked on top of already-running instances

**Fix — three changes applied together:**

**A. Second Spark worker** (`docker-compose.yaml`):
```yaml
spark-worker-2:
  build: ./spark
  command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
  depends_on:
    - spark-master
  environment:
    - SPARK_MODE=worker
    - SPARK_MASTER_URL=spark://spark-master:7077
    - SPARK_WORKER_MEMORY=3G
    - SPARK_WORKER_CORES=2
  volumes:
    - ~/.aws:/root/.aws:ro
```
Cluster now has 4 cores / 6 GB total. Both DAGs can run Spark jobs in parallel without blocking. Worker memory also bumped from 2 GB → 3 GB each.

**B. Staggered schedule + `max_active_runs=1`** (both DAGs):
```python
# daily_processing — stays at midnight
schedule_interval="0 0 * * *"
max_active_runs=1  # if still running, skip the next trigger

# ml_retraining — moved to 2am, every 3 days
schedule_interval="0 2 */3 * *"
max_active_runs=1
```
The 2-hour gap gives `daily_processing` time to finish before `ml_retraining` starts. `max_active_runs=1` ensures retries don't pile on top of a running instance.

**C. `ExternalTaskSensor` as first task in `ml_retraining`**:
```python
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_daily = ExternalTaskSensor(
    task_id="wait_for_daily_processing",
    external_dag_id="daily_processing",
    external_task_id="notify_success",
    allowed_states=["success"],
    execution_delta=timedelta(hours=2),
    timeout=7200,       # wait up to 2 hours before failing
    poke_interval=60,   # check every minute
    mode="reschedule",  # releases the worker slot while waiting — does not hold it
)

wait_for_daily >> validate >> prepare >> train >> notify
```
Even if `daily_processing` runs long, `ml_retraining` blocks safely without consuming a worker slot. If `daily_processing` fails, `ml_retraining` times out cleanly rather than training on stale data.

**Retries also reduced** — daily: 3→1, ml: 2→1. Fewer retries mean less queue buildup when something genuinely fails.

---

### 2.7 Airflow Executor Congestion From Stale Failed Runs

**Problem:** Multiple failed DAG runs queued up retries simultaneously. With SequentialExecutor this means runs pile up and block each other indefinitely — the scheduler never clears its in-memory queue even if you delete runs in the UI.

**Fix:** Clear failed runs programmatically and restart the scheduler:
```python
# Inside an airflow shell or Python session
from airflow.utils.session import create_session
from airflow.models import DagRun
from airflow.utils.state import State

with create_session() as session:
    runs = session.query(DagRun).filter(
        DagRun.dag_id == "ml_retraining",
        DagRun.state.in_([State.RUNNING, State.QUEUED])
    ).all()
    for r in runs:
        r.state = State.FAILED
    session.commit()
```
Then restart: `docker-compose restart airflow-scheduler`

---

### 2.8 Duplicate Dict Keys in PySpark `.agg()` Are Silently Dropped

**Problem:** `df.groupBy("symbol").agg({"price": "min", "price": "max", "price": "avg"})` only keeps the last key — Python silently overwrites duplicate dict keys.

**Fix:** Use explicit column expressions with `.alias()`:
```python
from pyspark.sql.functions import min as spark_min, max as spark_max, avg as spark_avg

df.groupBy("symbol").agg(
    spark_min("price").alias("min_price"),
    spark_max("price").alias("max_price"),
    spark_avg("price").alias("avg_price"),
)
```

---

### 2.9 Streamlit State Across Reruns Requires Module-Level Variables

**Problem:** Streamlit reruns the entire script on every refresh cycle. Any state stored inside functions or `st.session_state` is reset or unreliable when background threads are writing to it concurrently.

**Fix:** Use module-level `defaultdict(deque)` so state persists across reruns and background threads can write to it safely:
```python
from collections import defaultdict, deque

live_prices = defaultdict(lambda: deque(maxlen=200))
predictions = defaultdict(lambda: deque(maxlen=200))

# Background threads append to these — they persist across Streamlit reruns
```

---

### 2.10 Trailing Whitespace After Line Continuation Breaks Python

**Problem:** A single trailing space after `\` in a multi-line Python expression causes `SyntaxError: unexpected character after line continuation character`. This is invisible in most editors.

```python
# BROKEN — trailing space after \
spark = SparkSession.builder \  
    .appName("merge_raw") \
    .master("spark://spark-master:7077") \
    .getOrCreate()
```

**Fix:** Ensure no whitespace follows `\`. Use a linter or `rstrip()` check in CI.

---

## 3. Docker Command Reference

### Starting the Full Pipeline

```bash
# Start everything
docker-compose up -d

# Start specific services only
docker-compose up -d kafka producer producer2
docker-compose up -d consumer-prices consumer-orderbook
docker-compose up -d spark-master spark-worker spark-worker-2
docker-compose up -d airflow-webserver airflow-scheduler
docker-compose up -d predictor frontend
```

### Rebuilding After Code Changes

```bash
# Rebuild a single service
docker-compose build <service>
docker-compose up -d <service>

# Rebuild without cache (when dependencies change)
docker-compose build --no-cache <service>

# Rebuild the ML image (required before Airflow can retrain)
docker-compose build --no-cache ml

# Rebuild and restart predictor after model or code changes
docker-compose build predictor && docker-compose up -d predictor
```

### Viewing Logs

```bash
# Follow logs for a service
docker logs -f stock-pipeline-predictor-1

# All services (noisy)
docker-compose logs -f

# Filter predictor predictions only
docker logs stock-pipeline-predictor-1 2>&1 | grep -v "kafka\."

# Check if predictions are flowing
docker logs stock-pipeline-predictor-1 2>&1 | grep "\[BTC\|ETH\|SOL\]"
```

### Checking Container Status

```bash
# All running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check which services are down
docker-compose ps
```

### Stopping Services

```bash
# Stop everything (keeps volumes)
docker-compose down

# Stop everything and wipe volumes (fresh start)
docker-compose down -v

# Stop a single service
docker-compose stop predictor

# Restart a service
docker-compose restart airflow-scheduler
```

### Kafka Inspection

```bash
# List topics
docker exec stock-pipeline-kafka-1 kafka-topics --bootstrap-server localhost:9092 --list

# Check consumer group lag (predictor)
docker exec stock-pipeline-kafka-1 kafka-consumer-groups \
    --bootstrap-server localhost:9092 \
    --group predictor-prices \
    --describe

# Check latest offsets on crypto-predictions
docker exec stock-pipeline-kafka-1 kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 \
    --topic crypto-predictions
```

### Spark Inspection

```bash
# Spark master UI (shows both workers, active jobs, memory/core usage)
open http://localhost:8080

# Check both workers are registered with the master
docker logs stock-pipeline-spark-master-1 2>&1 | grep "Registering worker"

# Submit a job manually (from inside Airflow container)
docker exec stock-pipeline-airflow-webserver-1 \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /opt/airflow/spark/jobs/merge_raw.py

# Scale Spark workers up/down on the fly
docker-compose up -d --scale spark-worker=3   # adds a third worker
docker-compose up -d --scale spark-worker=1   # back to one worker
```

### ML Training Manual Trigger

```bash
# Run training manually outside Airflow
docker run --rm \
    -e AWS_DEFAULT_REGION=eu-north-1 \
    -v ~/.aws:/root/.aws:ro \
    stock-pipeline-ml \
    python train.py

# Run with custom args
docker run --rm \
    -e AWS_DEFAULT_REGION=eu-north-1 \
    -v ~/.aws:/root/.aws:ro \
    stock-pipeline-ml \
    python train.py --epochs 5
```

---

## 4. Airflow Operations

### Access
- URL: http://localhost:8081
- Username: `admin` / Password: `admin`

### Triggering DAGs

```bash
# Trigger daily_processing manually via CLI
docker exec stock-pipeline-airflow-webserver-1 \
    airflow dags trigger daily_processing

# Trigger ml_retraining manually
docker exec stock-pipeline-airflow-webserver-1 \
    airflow dags trigger ml_retraining
```

### Clearing Stuck Tasks

```bash
# Clear a specific task so it reruns
docker exec stock-pipeline-airflow-webserver-1 \
    airflow tasks clear ml_retraining -t train_lstm -y

# Mark a task as success to unblock the DAG
docker exec stock-pipeline-airflow-webserver-1 \
    airflow tasks states-for-dag-run ml_retraining <run_id>
```

### Killing Stale Runs (SQLite lock scenario)

```bash
docker exec -it stock-pipeline-airflow-webserver-1 python3 << 'EOF'
from airflow.utils.session import create_session
from airflow.models import DagRun
from airflow.utils.state import State

with create_session() as session:
    runs = session.query(DagRun).filter(
        DagRun.dag_id == "ml_retraining",
        DagRun.state.in_([State.RUNNING, State.QUEUED])
    ).all()
    for r in runs:
        print(f"Marking failed: {r.run_id}")
        r.state = State.FAILED
    session.commit()
EOF

docker-compose restart airflow-scheduler
```

### Checking Task Logs

```bash
# List recent DAG runs
docker exec stock-pipeline-airflow-webserver-1 \
    airflow dags list-runs -d ml_retraining --limit 5

# Get task instance state
docker exec stock-pipeline-airflow-webserver-1 \
    airflow tasks list ml_retraining
```

---

## 5. AWS & S3 Reference

### Secret Structure

Secret name: `stock-pipeline/spark` (AWS Secrets Manager, eu-north-1)
```json
{
  "S3_BUCKET": "<your-bucket-name>"
}
```

### S3 Bucket Layout

```
s3://<bucket>/
├── raw/
│   ├── prices/date=YYYY-MM-DD/          ← JSON trade records
│   └── orderbook/date=YYYY-MM-DD/       ← JSON orderbook snapshots
├── processed/
│   ├── merged/date=YYYY-MM-DD/          ← parquet, deduplicated
│   ├── joined/                          ← parquet, prices + orderbook joined
│   ├── daily/                           ← parquet, 15 engineered features
│   ├── training/                        ← parquet, cleaned training set
│   └── models/
│       ├── BTCUSDT/<timestamp>/
│       │   ├── model.keras
│       │   ├── feature_scaler.joblib
│       │   ├── price_scaler.joblib
│       │   └── metrics.json
│       ├── ETHUSDT/<timestamp>/
│       └── SOLUSDT/<timestamp>/
```

### Useful S3 Commands

```bash
# List all model versions
aws s3 ls s3://<bucket>/processed/models/ --recursive

# Check training data size
aws s3 ls s3://<bucket>/processed/training/ --recursive --human-readable

# Download a model locally
aws s3 cp s3://<bucket>/processed/models/BTCUSDT/<timestamp>/model.keras ./

# Check raw data volume by date
aws s3 ls s3://<bucket>/raw/prices/ --recursive --human-readable | tail -20
```

---

## 6. Environment Variables (.env)

All configuration is centralised in a single `.env` file at the project root. Docker Compose reads this automatically on every `docker-compose up`. The file is gitignored — only `.env.example` (with no real values) is committed to GitHub.

### `.env` Structure

```env
# AWS
AWS_DEFAULT_REGION=eu-north-1

# Kafka
KAFKA_CLUSTER_ID=MkU3OEVBNTcwNTJENDM2Qg

# AWS Secrets Manager — secret paths (not the secrets themselves)
SECRET_SPARK=stock-pipeline/spark
SECRET_PRODUCER=stock-pipeline/producer
SECRET_CONSUMER=stock-pipeline/consumer
SECRET_AIRFLOW=stock-pipeline/airflow
SECRET_ML=stock-pipeline/ml

# Airflow
AIRFLOW_ADMIN_PASSWORD=your_password_here
```

### How Each Variable Is Used

| Variable | Used By | Purpose |
|---|---|---|
| `AWS_DEFAULT_REGION` | all Python services | boto3 client region for S3 + Secrets Manager |
| `KAFKA_CLUSTER_ID` | docker-compose (Kafka) | KRaft mode cluster identity — must be stable across restarts |
| `SECRET_SPARK` | Spark jobs, predictor | Path to `{"S3_BUCKET": "..."}` in Secrets Manager |
| `SECRET_PRODUCER` | producer, producer2, predictor | Path to `{"KAFKA_BOOTSTRAP_SERVERS": "..."}` in Secrets Manager |
| `SECRET_CONSUMER` | consumer | Path to `{"KAFKA_BOOTSTRAP_SERVERS": "...", "S3_BUCKET": "..."}` in Secrets Manager |
| `SECRET_AIRFLOW` | Airflow DAGs | Path to `{"S3_BUCKET": "..."}` in Secrets Manager |
| `SECRET_ML` | ml/train.py | Path to `{"S3_BUCKET": "..."}` in Secrets Manager |
| `AIRFLOW_ADMIN_PASSWORD` | airflow-init | Sets the admin UI password on first init |

### What Is NOT in `.env` (stored in AWS Secrets Manager instead)

The actual values fetched at runtime — `S3_BUCKET`, `KAFKA_BOOTSTRAP_SERVERS` — live in AWS Secrets Manager. `.env` only holds the **path names** to those secrets, not the secrets themselves. This means `.env` is safe to share with teammates — it contains no credentials.

### Changing the AWS Region

To move to a different AWS region, update only one line:
```env
AWS_DEFAULT_REGION=us-east-1
```
Every Python file and docker-compose service reads this via `os.environ["AWS_DEFAULT_REGION"]` — no code changes needed.

---

## 7. Data Stream Reference

### The Two Binance WebSocket Streams

The pipeline uses two separate Binance WebSocket streams, one per producer:

#### producer — Trade Stream (`@trade`)
```
wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/solusdt@trade
```

| Field | Description |
|---|---|
| `e` | Event type (`trade`) |
| `s` | Symbol (`BTCUSDT`) |
| `p` | Trade price |
| `q` | Trade quantity (size) |
| `T` | Trade timestamp (ms) |
| `t` | Trade ID |

**What it gives you:** Every single executed trade in real time. Each message = one transaction between a buyer and seller. High frequency — BTC can produce hundreds per second.

**Published to:** `crypto-prices` Kafka topic

---

#### producer2 — Order Book Stream (`@depth20@1000ms`)
```
wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@1000ms/ethusdt@depth20@1000ms/solusdt@depth20@1000ms
```

| Field | Description |
|---|---|
| `bids` | Top 20 bid levels `[[price, qty], ...]` |
| `asks` | Top 20 ask levels `[[price, qty], ...]` |

**What it gives you:** A snapshot of the order book — the top 20 prices where people are willing to buy (bids) and sell (asks), updated every 1000ms. Not individual trades — this is pending intent.

**Computed fields added by producer2:**
- `top_bid_price` / `top_ask_price` — best available prices
- `spread` = `top_ask_price - top_bid_price` — cost of crossing the market
- `bid_ask_ratio` = `total_bid_volume / total_ask_volume` — buy vs sell pressure
- `total_bid_volume` / `total_ask_volume` — aggregated depth

**Published to:** `crypto-orderbook` Kafka topic

---

### Why Two Separate Streams?

| | Trade Stream | Order Book Stream |
|---|---|---|
| **Frequency** | Every trade (real-time) | Every 1 second (snapshot) |
| **Nature** | What DID happen | What MIGHT happen |
| **Signal** | Actual price, volume executed | Market depth, liquidity, intent |
| **Use in ML** | Price lags, rolling averages, momentum | Spread, bid/ask ratio, liquidity features |

Combining both gives the LSTM model **15 features** that capture both executed market activity and the underlying order book structure — neither stream alone is sufficient for a complete picture.

---

### Why Binance?

- Free, no API key required
- Highest crypto liquidity globally — tightest spreads, most trades per second
- Stable WebSocket with reliable reconnect behaviour
- Provides both trade and order book streams for BTC/ETH/SOL

---

## 8. Data Contracts

Data contracts enforce a formal schema agreement between producers and consumers. If a producer changes the shape of a message, the contract fails loudly rather than silently breaking downstream Spark jobs, ML training, or inference.

The pipeline implements two levels of contracts:

---

### 8.1 Option A — Pydantic Contracts at the Kafka Boundary

**Location:** `contracts/trade.py`, `contracts/orderbook.py`

These contracts live in a shared `contracts/` folder copied into every producer and consumer Docker image at build time.

#### `TradeEvent` — `contracts/trade.py`

Validates every message before it is published to `crypto-prices`:

| Field | Type | Rule |
|---|---|---|
| `symbol` | `str` | Must be one of `BTCUSDT`, `ETHUSDT`, `SOLUSDT` |
| `price` | `float` | Must be > 0 |
| `size` | `float` | Must be > 0 |
| `timestamp` | `int` | Must be > 1,577,836,800,000 (after Jan 2020 in ms) |
| `trade_id` | `int` | Required |

#### `OrderbookEvent` — `contracts/orderbook.py`

Validates every message before it is published to `crypto-orderbook`:

| Field | Type | Rule |
|---|---|---|
| `symbol` | `str` | Must be one of `BTCUSDT`, `ETHUSDT`, `SOLUSDT` |
| `top_bid_price` | `float` | Must be > 0 |
| `top_ask_price` | `float` | Must be > 0 **and** > `top_bid_price` |
| `total_bid_volume` | `float` | Must be > 0 |
| `total_ask_volume` | `float` | Must be > 0 |
| `bid_ask_ratio` | `float` | Must be > 0 |
| `timestamp` | `int` | Must be > 1,577,836,800,000 ms |

#### How violations are handled

**Producer (publish-side):**
```python
try:
    validated = TradeEvent(**raw)
    payload = validated.model_dump()
except ValidationError as e:
    logger.error(f"[CONTRACT VIOLATION] TradeEvent: {e} — sending to DLQ")
    send_with_retry(DLQ_TOPIC, {"error": str(e), "raw": raw})
    return  # do not publish to main topic
```
Violations are routed to the dead-letter queue (`crypto-prices-dlq` / `crypto-orderbook-dlq`) and never reach the main topic.

**Consumer (read-side):**
```python
contract = CONTRACTS.get(TOPIC)  # TradeEvent or OrderbookEvent

for message in consumer:
    try:
        contract(**record)
    except ValidationError as e:
        logger.error(f"[CONTRACT VIOLATION] {TOPIC}: {e} — skipping record")
        continue  # do not write to S3
    batch.append(record)
```
Any message that slips past the producer (e.g. replayed from an old offset) is re-validated on the consumer side. Violations are skipped — never written to S3.

#### Docker build context

Because `contracts/` is shared across multiple services, the build context was changed from the individual service folder to the project root:

```yaml
producer:
  build:
    context: .                    # project root
    dockerfile: producer/dockerfile
```

The Dockerfile then copies both the service code and the contracts:
```dockerfile
COPY producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY producer/producer.py .
COPY contracts ./contracts        # shared contract models
```

---

### 8.2 Option B — Data Quality Checks Between Spark Stages

Schema and quality checks are wired as Airflow `PythonOperator` tasks that gate each Spark job. If a check fails, the DAG stops before the next stage runs.

#### `daily_processing` DAG

```
validate_raw → merge_raw → check_merged → join_topics → feature_engineering → check_features → notify
```

| Task | What it checks |
|---|---|
| `check_merged_data` | Confirms `processed/merged/` has parquet files after `merge_raw` |
| `check_features_data` | Confirms `processed/daily/` exists AND reads parquet schema via pyarrow to verify all 15 feature columns are present |

**Feature column contract** — `check_features_data` will fail the DAG if any of these are missing:
```
symbol, price, size,
price_lag_1, price_lag_5, price_lag_10,
price_change, price_change_pct,
rolling_avg_5, rolling_avg_10, rolling_stddev_10,
volume_change, spread, bid_ask_ratio,
total_bid_volume, total_ask_volume
```

#### `ml_retraining` DAG

```
wait_for_daily → validate → prepare_training → check_training → train_lstm → notify
```

| Task | What it checks |
|---|---|
| `check_training_data` | Confirms `processed/training/` has files and total size > 1 KB — prevents training on empty or corrupted output |

#### Why gate before training specifically?

Training is the most expensive task (~3 minutes, holds the SQLite lock). If `prepare_training` silently produced empty output, training would either crash mid-run or produce a useless model. `check_training` catches this in seconds before the LSTM starts.

---

### 8.3 What Happens When a Contract Is Violated

| Violation point | Behaviour |
|---|---|
| Producer publishes invalid trade | Routed to `crypto-prices-dlq`, logged as `[CONTRACT VIOLATION]` |
| Producer publishes invalid orderbook | Routed to `crypto-orderbook-dlq`, logged as `[CONTRACT VIOLATION]` |
| Consumer reads invalid message | Routed to `crypto-prices-dlq` / `crypto-orderbook-dlq` — preserved for inspection, not written to S3 |
| Merged layer empty after merge_raw | `check_merged_data` raises `ValueError` — DAG fails, join does not run |
| Feature column missing in parquet | `check_features_data` raises `ValueError` — DAG fails, downstream tasks do not run |
| Training data empty or tiny | `check_training_data` raises `ValueError` — `train_lstm` does not run |

The DLQ consumers (`consumer-prices-dlq`, `consumer-orderbook-dlq`) pick up everything routed to the dead-letter topics and write it to `s3://<bucket>/errors/producer/` for later inspection or replay.

#### Testing contract violations manually

Use these commands to inject bad messages and watch the DLQ catch them:

```bash
# Bad symbol — not in {BTCUSDT, ETHUSDT, SOLUSDT}
echo '{"symbol": "XYZUSDT", "price": 100.0, "size": 0.5, "timestamp": 1775667646272, "trade_id": 9999}' | \
  docker exec -i stock-pipeline-kafka-1 kafka-console-producer \
  --bootstrap-server localhost:9092 --topic crypto-prices

# Negative price
echo '{"symbol": "BTCUSDT", "price": -50.0, "size": 0.5, "timestamp": 1775667646272, "trade_id": 9998}' | \
  docker exec -i stock-pipeline-kafka-1 kafka-console-producer \
  --bootstrap-server localhost:9092 --topic crypto-prices

# Bad timestamp (too old — not in milliseconds)
echo '{"symbol": "BTCUSDT", "price": 71000.0, "size": 0.5, "timestamp": 999, "trade_id": 9997}' | \
  docker exec -i stock-pipeline-kafka-1 kafka-console-producer \
  --bootstrap-server localhost:9092 --topic crypto-prices
```

Then watch the consumer catch and route them:
```bash
# Consumer logs — should show [CONTRACT VIOLATION] → routing to crypto-prices-dlq
docker logs stock-pipeline-consumer-prices-1 2>&1 | grep "CONTRACT"

# DLQ consumer logs — should show the batch written to S3 errors/
docker logs stock-pipeline-consumer-prices-dlq-1 2>&1 | grep "Written"
```

---

### 8.4 Extending Contracts

**To add a new symbol** (e.g. `SOLUSDT` → add `BNBUSDT`):
1. Add `"BNBUSDT"` to `VALID_SYMBOLS` in both `contracts/trade.py` and `contracts/orderbook.py`
2. Add the symbol to the Binance stream URLs in `producer/producer.py` and `producer2/producer2.py`
3. Rebuild producer, producer2, consumer images

**To add a new feature column:**
1. Add the computation in `spark/jobs/feature_engineering.py`
2. Add the column name to `REQUIRED_FEATURE_COLS` in `check_features_data` in `airflow/dags/daily_processing_dag.py`
3. Add the column to `FEATURE_COLS` in `predictor/predictor.py`
4. Retrain the model

---

## 9. Production Way Forward

The current setup is a working prototype. Below is the recommended path to make it production-grade.

---

### 6.1 Replace SQLite + SequentialExecutor With PostgreSQL + Celery

**Current limitation:** SQLite locks under long tasks; SequentialExecutor runs one task at a time. The DAG collision between `daily_processing` and `ml_retraining` is currently mitigated with a second Spark worker, staggered schedules, `max_active_runs=1`, and an `ExternalTaskSensor` — but the underlying SQLite single-writer bottleneck remains and will resurface as more DAGs or longer tasks are added.

```yaml
# docker-compose addition
postgres:
  image: postgres:15
  environment:
    POSTGRES_DB: airflow
    POSTGRES_USER: airflow
    POSTGRES_PASSWORD: airflow

redis:
  image: redis:7

airflow-worker:
  build: ./airflow
  command: airflow celery worker
  environment:
    - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
    - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
    - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
```

This allows parallel task execution and eliminates the database lock issue entirely.

---

### 6.2 Add MLflow for Model Versioning and Hot-Reload

**Current limitation:** The predictor loads the latest model at startup only — a new trained model requires a container restart.

**Production pattern:**
```
train.py → log model to MLflow registry (stage: Production)
predictor.py → poll MLflow every N minutes for new Production models
             → hot-reload without restart
```

```python
# In train.py
import mlflow
mlflow.keras.log_model(model, "lstm_model", registered_model_name=f"lstm_{symbol}")
mlflow.sklearn.log_model(feature_scaler, "feature_scaler")

# In predictor.py
client = mlflow.tracking.MlflowClient()
latest = client.get_latest_versions(f"lstm_{symbol}", stages=["Production"])[0]
model = mlflow.keras.load_model(latest.source)
```

---

### 6.3 Add Kafka Partitioning and Multiple Workers

**Current limitation:** Single partition per topic — only one consumer instance can read per group.

```bash
# Create topics with 3 partitions (one per symbol)
kafka-topics --create --topic crypto-prices \
    --partitions 3 --replication-factor 1 \
    --bootstrap-server localhost:9092
```

- Route BTC → partition 0, ETH → partition 1, SOL → partition 2 (key-based)
- Scale to 3 consumer instances and 3 predictor instances — one per symbol
- Each predictor only processes its symbol's partition

---

### 6.4 Add Grafana + Prometheus for Pipeline Monitoring

**Metrics to track:**
- Kafka consumer lag per topic/group
- Prediction latency (time from trade received to prediction published)
- Model inference time per symbol
- S3 write throughput (records/second)
- Spark job duration per DAG run
- Airflow task success/failure rates

```yaml
# docker-compose addition
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

Use the **Kafka Exporter** (`danielqsj/kafka-exporter`) and **JMX Exporter** for Spark metrics.

---

### 6.5 Replace Standalone Spark With EMR or Databricks

**Current limitation:** Spark runs on a single Docker host — no true distributed compute.

**Options:**
- **AWS EMR Serverless** — submit PySpark jobs without managing a cluster; pay per job
- **AWS Glue** — fully managed Spark with native S3 integration; no cluster management
- **Databricks** — best for iterative ML + Spark; native MLflow integration

For this pipeline, EMR Serverless is the most cost-efficient path — trigger jobs from Airflow using `EmrServerlessOperator`.

---

### 6.6 Move Secrets Out of Environment Variables

**Current:** AWS credentials mounted from `~/.aws` via Docker volume.

**Production:**
- Use **AWS IAM Roles** attached to ECS tasks or EC2 instances — no credential files needed
- Store `POLYGON_API_KEY` and other secrets in AWS Secrets Manager
- Inject secrets at runtime via the AWS SDK, not environment variables or files

---

### 6.7 Add Data Quality Checks Before Training

**Current limitation:** `prepare_training.py` drops nulls and fills with 0 — no validation that the data is sane.

**Add Great Expectations or simple assertion checks:**
```python
# Minimum records per symbol
assert df.filter(col("symbol") == "BTCUSDT").count() >= 1000, "Insufficient BTC data"

# Price sanity range
assert df.filter((col("price") < 1) | (col("price") > 1_000_000)).count() == 0

# No all-zero feature rows
assert df.filter(col("rolling_avg_5") == 0).count() / df.count() < 0.01
```

Fail the Airflow DAG task if checks fail — don't train on bad data.

---

### 6.8 LSTM Model Improvements for Better Predictions

**Current model:** 2 LSTM layers, 2 epochs, 10k records/symbol, next-trade price prediction.

**Improvements for a more useful signal:**

| Change | Impact |
|---|---|
| Increase epochs to 20–50 | Better convergence — requires PostgreSQL Airflow first |
| Predict price 5 or 10 trades ahead | More actionable signal |
| Add attention mechanism | Lets model weight recent trades more |
| Per-symbol hyperparameter tuning | BTC volatility differs from SOL |
| Use returns (% change) as target instead of raw price | Model generalises better across price ranges |
| Walk-forward validation | Prevents lookahead bias in evaluation |

---

### 6.9 Production Deployment Target

For full production deployment the recommended target is:

```
AWS ECS Fargate
  ├── kafka (or MSK — managed Kafka)
  ├── producer / producer2
  ├── consumer-prices / consumer-orderbook
  ├── predictor (one task per symbol)
  └── frontend

AWS EMR Serverless
  └── daily Spark batch jobs (triggered by Airflow MWAA)

AWS MWAA (Managed Airflow)
  └── daily_processing + ml_retraining DAGs

AWS SageMaker
  └── LSTM training + model registry

AWS CloudWatch + Grafana Cloud
  └── pipeline monitoring and alerting
```

This eliminates all self-managed infrastructure while keeping the same logical architecture built here.
