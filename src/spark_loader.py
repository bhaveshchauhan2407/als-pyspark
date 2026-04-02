from typing import List, Tuple
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os


def create_spark_session():
    os.environ["PYSPARK_PYTHON"] = "python"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

    spark = (
        SparkSession.builder
        .appName("FastALSProject")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_yelp_interactions_spark(spark, path: str, limit_rows: int = 5000):
    raw_df = spark.read.text(path)
    parts = F.split(F.trim(F.col("value")), r"\s+")

    df = raw_df.select(
        parts.getItem(0).cast("int").alias("user_id"),
        parts.getItem(1).cast("int").alias("item_id"),
        parts.getItem(2).cast("float").alias("rating"),
        parts.getItem(3).cast("long").alias("timestamp"),
    )

    df = (
        df.select("user_id", "item_id")
          .dropna()
          .dropDuplicates()
          .limit(limit_rows)
    )

    return df


def load_yelp_events_spark(spark, path: str, limit_rows: int = 5000):
    """
    For online protocol:
    keep timestamp and do NOT drop duplicates.
    """
    raw_df = spark.read.text(path)
    parts = F.split(F.trim(F.col("value")), r"\s+")

    df = raw_df.select(
        parts.getItem(0).cast("int").alias("user_id"),
        parts.getItem(1).cast("int").alias("item_id"),
        parts.getItem(2).cast("float").alias("rating"),
        parts.getItem(3).cast("long").alias("timestamp"),
    ).dropna()

    df = df.orderBy("timestamp").limit(limit_rows)
    return df


def spark_df_to_interactions(df) -> List[Tuple[int, int]]:
    rows = df.collect()
    return [(row["user_id"], row["item_id"]) for row in rows]


def spark_df_to_events(df) -> List[Tuple[int, int, int]]:
    rows = df.collect()
    return [(row["user_id"], row["item_id"], row["timestamp"]) for row in rows]


def build_groupings_with_spark(df):
    user_items_df = df.groupBy("user_id").agg(F.collect_list("item_id").alias("item_list"))
    item_users_df = df.groupBy("item_id").agg(F.collect_list("user_id").alias("user_list"))
    return user_items_df, item_users_df