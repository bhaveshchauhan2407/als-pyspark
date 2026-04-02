from typing import List, Tuple


def spark_df_to_rdd(df):

    return df.rdd.map(lambda row: (row["user_id"], row["item_id"]))


def build_user_items_rdd(interactions_rdd):

    return interactions_rdd.groupByKey().mapValues(list)


def build_item_users_rdd(interactions_rdd):

    return interactions_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)


def collect_rdd_interactions(interactions_rdd) -> List[Tuple[int, int]]:

    return interactions_rdd.collect()