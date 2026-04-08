
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# import argparse
# import json
# import boto3
# from datetime import datetime, timedelta

# def get_secret(secret_name: str) -> dict:
#     client = boto3.client("secretsmanager", region_name="eu-north-1")
#     response = client.get_secret_value(SecretId=secret_name)
#     return json.loads(response["SecretString"])

# parser = argparse.ArgumentParser()
# parser.add_argument("--days", type=int, default=90)
# args = parser.parse_args()

# config = get_secret(os.environ["SECRET_SPARK"])
# S3_BUCKET = config["S3_BUCKET"]

# spark = SparkSession.builder \
#     .appName("prepare_training") \
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
#     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
#     .getOrCreate()

# input_path = f"s3a://{S3_BUCKET}/processed/daily/"
# df = spark.read.option("recursiveFileLookup", "true").parquet(input_path)

# feature_cols = [
#     "price", "size",
#     "price_lag_1", "price_lag_5", "price_lag_10",
#     "price_change", "price_change_pct",
#     "rolling_avg_5", "rolling_avg_10",
#     "rolling_stddev_10", "volume_change",
#     "spread", "bid_ask_ratio",
#     "total_bid_volume", "total_ask_volume",
# ]

# df = df.select(["symbol", "timestamp"] + feature_cols)

# # Only drop rows where core price features are null
# df = df.dropna(subset=["price", "price_lag_1", "price_change", "rolling_avg_5"])

# # Fill remaining nulls with 0
# df = df.fillna(0, subset=[
#     "bid_ask_ratio", "total_bid_volume", "total_ask_volume",
#     "spread"
# ])

# output_path = f"s3a://{S3_BUCKET}/processed/training/"
# df.write.mode("overwrite").parquet(output_path)
# print(f"Prepared {df.count()} records for LSTM training")

# spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, min as spark_min, max as spark_max, avg as spark_avg, row_number, desc
from pyspark.sql.window import Window
import argparse
import json
import os
import boto3

def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=90)
parser.add_argument("--limit-per-symbol", type=int, default=10000)
args = parser.parse_args()

config = get_secret(os.environ["SECRET_SPARK"])
S3_BUCKET = config["S3_BUCKET"]

spark = SparkSession.builder \
    .appName("prepare_training") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

input_path = f"s3a://{S3_BUCKET}/processed/daily/"
df = spark.read.option("recursiveFileLookup", "true").parquet(input_path)

print(f"\nTotal records loaded: {df.count()}")
print("Symbol distribution:")
df.groupBy("symbol").count().orderBy("symbol").show()

feature_cols = [
    "price", "size",
    "price_lag_1", "price_lag_5", "price_lag_10",
    "price_change", "price_change_pct",
    "rolling_avg_5", "rolling_avg_10",
    "rolling_stddev_10", "volume_change",
    "spread", "bid_ask_ratio",
    "total_bid_volume", "total_ask_volume",
]

df = df.select(["symbol", "timestamp"] + feature_cols)

# Limit records per symbol (most recent) to keep training fast
w = Window.partitionBy("symbol").orderBy(desc("timestamp"))
df = df.withColumn("_row", row_number().over(w)) \
       .filter(col("_row") <= args.limit_per_symbol) \
       .drop("_row")
print(f"\nAfter limiting to {args.limit_per_symbol} records per symbol:")
df.groupBy("symbol").count().orderBy("symbol").show()

# Drop rows where core price features are null
before = df.count()
df = df.dropna(subset=["price", "price_lag_1", "price_change", "rolling_avg_5"])
after = df.count()
print(f"\nRows dropped by dropna: {before - after}")

# --- Check nulls BEFORE fillna ---
print("\nNaN counts BEFORE fillna:")
nan_counts_before = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in df.columns
    if c != "symbol"
]).collect()[0].asDict()

for col_name, nan_count in nan_counts_before.items():
    if nan_count > 0:
        print(f"  {col_name}: {nan_count} NaN/null values ⚠️")
    else:
        print(f"  {col_name}: clean ✅")

# Fill remaining nulls with 0
df = df.fillna(0, subset=[
    "bid_ask_ratio", "total_bid_volume", "total_ask_volume", "spread"
])

# --- Check nulls AFTER fillna ---
print("\nNaN counts AFTER fillna:")
nan_counts_after = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in df.columns
    if c != "symbol"
]).collect()[0].asDict()

for col_name, nan_count in nan_counts_after.items():
    if nan_count > 0:
        print(f"  {col_name}: {nan_count} NaN/null values ⚠️")
    else:
        print(f"  {col_name}: clean ✅")

# Price range per symbol
print("\nPrice range per symbol:")
df.groupBy("symbol").agg(
    spark_min("price").alias("min_price"),
    spark_max("price").alias("max_price"),
    spark_avg("price").alias("avg_price"),
).orderBy("symbol").show()

output_path = f"s3a://{S3_BUCKET}/processed/training/"
df.write.mode("overwrite").parquet(output_path)
print(f"\n✅ Prepared {df.count()} records for LSTM training")

spark.stop()