



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
#     .appName("merge_raw") \
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
#     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
#     .getOrCreate()

# for topic in ["crypto-prices", "crypto-orderbook"]:
#     input_path = f"s3a://{S3_BUCKET}/raw/{topic}/{year}/{month}/{day}/*.json"
#     output_path = f"s3a://{S3_BUCKET}/processed/merged/{topic}/{year}/{month}/{day}/"

#     df = spark.read.json(input_path)
#     df = df.dropDuplicates()
#     df.write.mode("overwrite").parquet(output_path)
#     print(f"Merged {df.count()} records for {topic} on {args.date}")

# spark.stop()




from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
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
    print(f"{'='*50}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True)
args = parser.parse_args()

config = get_secret(os.environ["SECRET_SPARK"])
S3_BUCKET = config["S3_BUCKET"]
year, month, day = args.date.split("-")

spark = SparkSession.builder \
    .appName("merge_raw") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

for topic in ["crypto-prices", "crypto-orderbook"]:
    print(f"\nProcessing topic: {topic}")
    input_path = f"s3a://{S3_BUCKET}/raw/{topic}/{year}/{month}/{day}/*.json"
    output_path = f"s3a://{S3_BUCKET}/processed/merged/{topic}/{year}/{month}/{day}/"

    # Read
    df = spark.read.json(input_path)
    validate_df(df, f"RAW READ — {topic}")

    # Deduplicate
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    print(f"Duplicates removed: {before - after}")

    validate_df(df, f"AFTER DEDUP — {topic}")

    # Write
    df.write.mode("overwrite").parquet(output_path)
    print(f"✅ Merged {after} records for {topic} on {args.date}")

spark.stop()