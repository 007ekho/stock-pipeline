


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, avg, stddev, lag, when
# from pyspark.sql.functions import count, when, isnan
# from pyspark.sql.window import Window
# import argparse
# import json
# import boto3

# def get_secret(secret_name: str) -> dict:
#     client = boto3.client("secretsmanager", region_name="eu-north-1")
#     response = client.get_secret_value(SecretId=secret_name)
#     return json.loads(response["SecretString"])

# parser = argparse.ArgumentParser()
# parser.add_argument("--date", required=True)
# args = parser.parse_args()

# config = get_secret("stock-pipeline/spark")
# S3_BUCKET = config["S3_BUCKET"]
# year, month, day = args.date.split("-")

# spark = SparkSession.builder \
#     .appName("feature_engineering") \
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
#     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
#     .getOrCreate()

# input_path = f"s3a://{S3_BUCKET}/processed/joined/{year}/{month}/{day}/"
# df = spark.read.parquet(input_path)

# w = Window.partitionBy("symbol").orderBy("timestamp")

# df = df \
#     .withColumn("price_lag_1", lag("price", 1).over(w)) \
#     .withColumn("price_lag_5", lag("price", 5).over(w)) \
#     .withColumn("price_lag_10", lag("price", 10).over(w)) \
#     .withColumn("price_change", col("price") - col("price_lag_1")) \
#     .withColumn("price_change_pct", (col("price") - col("price_lag_1")) / col("price_lag_1") * 100) \
#     .withColumn("rolling_avg_5", avg("price").over(w.rowsBetween(-5, 0))) \
#     .withColumn("rolling_avg_10", avg("price").over(w.rowsBetween(-10, 0))) \
#     .withColumn("rolling_stddev_10", stddev("price").over(w.rowsBetween(-10, 0))) \
#     .withColumn("volume_change", col("size") - lag("size", 1).over(w)) \
#     .withColumn("spread", col("top_ask_price") - col("top_bid_price")) \
#     .withColumn("market_pressure", when(col("bid_ask_ratio") > 1, "buy").otherwise("sell")) \
#     .dropna(subset=["price_lag_1", "price_change", "rolling_avg_5"])  # only drop if core cols are null

# nan_counts = df.select([
#     count(when(col(c).isNull() | isnan(c), c)).alias(c)
#     for c in df.columns
#     if c not in ["symbol", "market_pressure"]
# ]).collect()[0].asDict()

# for col_name, nan_count in nan_counts.items():
#     if nan_count > 0:
#         print(f"  {col_name}: {nan_count} NaN/null values")

# output_path = f"s3a://{S3_BUCKET}/processed/daily/{year}/{month}/{day}/"
# df.write.mode("overwrite").parquet(output_path)
# print(f"Feature engineered {df.count()} records for {args.date}")

# spark.stop()


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, lag, when, count, isnan
from pyspark.sql.window import Window
import argparse
import json
import boto3

def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name="eu-north-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True)
args = parser.parse_args()

config = get_secret("stock-pipeline/spark")
S3_BUCKET = config["S3_BUCKET"]
year, month, day = args.date.split("-")

spark = SparkSession.builder \
    .appName("feature_engineering") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

input_path = f"s3a://{S3_BUCKET}/processed/joined/{year}/{month}/{day}/"
df = spark.read.parquet(input_path)

w = Window.partitionBy("symbol").orderBy("timestamp")

df = df \
    .withColumn("price_lag_1", lag("price", 1).over(w)) \
    .withColumn("price_lag_5", lag("price", 5).over(w)) \
    .withColumn("price_lag_10", lag("price", 10).over(w)) \
    .withColumn("price_change",
        when(col("price_lag_1").isNull(), 0.0)
        .otherwise(col("price") - col("price_lag_1"))
    ) \
    .withColumn("price_change_pct",
        when(col("price_lag_1").isNull() | (col("price_lag_1") == 0), 0.0)
        .otherwise((col("price") - col("price_lag_1")) / col("price_lag_1") * 100)
    ) \
    .withColumn("rolling_avg_5", avg("price").over(w.rowsBetween(-5, 0))) \
    .withColumn("rolling_avg_10", avg("price").over(w.rowsBetween(-10, 0))) \
    .withColumn("rolling_stddev_10",
        when(stddev("price").over(w.rowsBetween(-10, 0)).isNull(), 0.0)
        .otherwise(stddev("price").over(w.rowsBetween(-10, 0)))
    ) \
    .withColumn("volume_change",
        when(lag("size", 1).over(w).isNull(), 0.0)
        .otherwise(col("size") - lag("size", 1).over(w))
    ) \
    .withColumn("spread",
        when(col("top_ask_price").isNull() | col("top_bid_price").isNull(), 0.0)
        .otherwise(col("top_ask_price") - col("top_bid_price"))
    ) \
    .withColumn("bid_ask_ratio",
        when(col("bid_ask_ratio").isNull(), 0.0)
        .otherwise(col("bid_ask_ratio"))
    ) \
    .withColumn("total_bid_volume",
        when(col("total_bid_volume").isNull(), 0.0)
        .otherwise(col("total_bid_volume"))
    ) \
    .withColumn("total_ask_volume",
        when(col("total_ask_volume").isNull(), 0.0)
        .otherwise(col("total_ask_volume"))
    ) \
    .withColumn("market_pressure",
        when(col("bid_ask_ratio") > 1, "buy").otherwise("sell")
    ) \
    .dropna(subset=["price_lag_1", "price_lag_5", "price_lag_10"])

# --- Debug NaN counts ---
print("\nNaN counts after feature engineering:")
nan_counts = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in df.columns
    if c not in ["symbol", "market_pressure"]
]).collect()[0].asDict()

for col_name, nan_count in nan_counts.items():
    if nan_count > 0:
        print(f"  {col_name}: {nan_count} NaN/null values")
    else:
        print(f"  {col_name}: clean ✅")

output_path = f"s3a://{S3_BUCKET}/processed/daily/{year}/{month}/{day}/"
df.write.mode("overwrite").parquet(output_path)
print(f"\nFeature engineered {df.count()} records for {args.date}")

spark.stop()