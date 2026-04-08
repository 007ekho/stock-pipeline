
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
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

# config = get_secret(os.environ["SECRET_SPARK"])
# S3_BUCKET = config["S3_BUCKET"]
# year, month, day = args.date.split("-")

# spark = SparkSession.builder \
#     .appName("join_topics") \
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
#     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
#     .getOrCreate()

# prices_path = f"s3a://{S3_BUCKET}/processed/merged/crypto-prices/{year}/{month}/{day}/"
# orderbook_path = f"s3a://{S3_BUCKET}/processed/merged/crypto-orderbook/{year}/{month}/{day}/"

# prices = spark.read.parquet(prices_path)
# orderbook = spark.read.parquet(orderbook_path)

# prices = prices.withColumn(
#     "timestamp_s",
#     (col("timestamp") / 1000).cast("long")
# )
# orderbook = orderbook.withColumn(
#     "timestamp_s",
#     (col("timestamp") / 1000).cast("long")
# )

# joined = prices.join(
#     orderbook,
#     on=[
#         prices.symbol == orderbook.symbol,
#         (prices.timestamp_s - orderbook.timestamp_s).between(-1, 1)
#     ],
#     how="left"
# ).select(
#     prices.symbol,
#     prices.timestamp,
#     prices.price,
#     prices.size,
#     orderbook.top_bid_price,
#     orderbook.top_ask_price,
#     orderbook.bid_ask_ratio,
#     orderbook.total_bid_volume,
#     orderbook.total_ask_volume,
# )

# output_path = f"s3a://{S3_BUCKET}/processed/joined/{year}/{month}/{day}/"
# joined.write.mode("overwrite").parquet(output_path)
# print(f"Joined {joined.count()} records for {args.date}")

# spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
import argparse
import json
import os
import boto3

def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

def validate_df(df, stage: str):
    print(f"\n{'='*50}")
    print(f"VALIDATION: {stage}")
    print(f"{'='*50}")
    print(f"Row count: {df.count()}")
    print(f"Schema:")
    df.printSchema()
    print(f"Sample data:")
    df.show(5, truncate=False)
    print(f"Null counts:")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    print(f"Symbol distribution:")
    df.groupBy("symbol").count().show()
    print(f"{'='*50}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True)
args = parser.parse_args()

config = get_secret(os.environ["SECRET_SPARK"])
S3_BUCKET = config["S3_BUCKET"]
year, month, day = args.date.split("-")

spark = SparkSession.builder \
    .appName("join_topics") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

prices_path = f"s3a://{S3_BUCKET}/processed/merged/crypto-prices/{year}/{month}/{day}/"
orderbook_path = f"s3a://{S3_BUCKET}/processed/merged/crypto-orderbook/{year}/{month}/{day}/"

prices = spark.read.parquet(prices_path)
orderbook = spark.read.parquet(orderbook_path)

validate_df(prices, "PRICES INPUT")
validate_df(orderbook, "ORDERBOOK INPUT")

# Convert timestamps
prices = prices.withColumn("timestamp_s", (col("timestamp") / 1000).cast("long"))
orderbook = orderbook.withColumn("timestamp_s", (col("timestamp") / 1000).cast("long"))

# Join
joined = prices.join(
    orderbook,
    on=[
        prices.symbol == orderbook.symbol,
        (prices.timestamp_s - orderbook.timestamp_s).between(-1, 1)
    ],
    how="left"
).select(
    prices.symbol,
    prices.timestamp,
    prices.price,
    prices.size,
    orderbook.top_bid_price,
    orderbook.top_ask_price,
    orderbook.bid_ask_ratio,
    orderbook.total_bid_volume,
    orderbook.total_ask_volume,
)

validate_df(joined, "AFTER JOIN")

# Check join quality
total = joined.count()
matched = joined.filter(col("top_bid_price").isNotNull()).count()
unmatched = total - matched
print(f"Join quality — matched: {matched}/{total} ({matched/total*100:.1f}%)")
print(f"Unmatched (no orderbook): {unmatched}/{total} ({unmatched/total*100:.1f}%)")

output_path = f"s3a://{S3_BUCKET}/processed/joined/{year}/{month}/{day}/"
joined.write.mode("overwrite").parquet(output_path)
print(f"✅ Joined {total} records for {args.date}")

spark.stop()