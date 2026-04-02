from typing import List, Tuple


def spark_df_to_rdd(df):
    """
    Convert Spark DataFrame with columns (user_id, item_id)
    into an RDD of (user_id, item_id).
    """
    return df.rdd.map(lambda row: (row["user_id"], row["item_id"]))


def build_user_items_rdd(interactions_rdd):
    """
    RDD: user_id -> list of item_id
    """
    return interactions_rdd.groupByKey().mapValues(list)


def build_item_users_rdd(interactions_rdd):
    """
    RDD: item_id -> list of user_id
    """
    return interactions_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)


def collect_rdd_interactions(interactions_rdd) -> List[Tuple[int, int]]:
    """
    Collect RDD back to local Python list for the current local training model.
    """
    return interactions_rdd.collect()