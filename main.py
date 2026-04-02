from src.config import FastALSConfig
from src.model import FastALSModel
from src.train import train_model
from src.utils import print_header
from src.spark_loader import (
    create_spark_session,
    load_yelp_interactions_spark,
    build_groupings_with_spark,
)
from src.rdd_loader import (
    spark_df_to_rdd,
    build_user_items_rdd,
    build_item_users_rdd,
    collect_rdd_interactions,
)
from src.split import leave_one_out_split
from src.evaluate import hit_rate_at_k
from src.recommender import recommend_top_k


def main():
    print_header("FastALS Python + Spark DF + RDD")

    config = FastALSConfig(
        factors=8,
        max_iter=3,
        reg=0.1,
        w0=1.0,
        alpha=0.5,
        init_mean=0.0,
        init_stdev=0.001,
        show_progress=True,
        show_loss=True,
        top_k=10,
    )

    data_path = "data/yelp.rating"

    spark = create_spark_session()

    df = load_yelp_interactions_spark(spark, data_path, limit_rows=10000)

    print("Spark DataFrame preview:")
    df.show(5)

    print(f"Spark interaction count: {df.count()}")

    user_items_df, item_users_df = build_groupings_with_spark(df)

    print("Grouped by user (DF) preview:")
    user_items_df.show(3, truncate=False)

    interactions_rdd = spark_df_to_rdd(df)

    print("RDD preview:")
    print(interactions_rdd.take(5))

    user_items_rdd = build_user_items_rdd(interactions_rdd)
    item_users_rdd = build_item_users_rdd(interactions_rdd)

    print("Grouped by user (RDD) preview:")
    print(user_items_rdd.take(3))

    print("Grouped by item (RDD) preview:")
    print(item_users_rdd.take(3))

    interactions = collect_rdd_interactions(interactions_rdd)
    print(f"Collected {len(interactions)} interactions into Python")

    train_interactions, test_interactions = leave_one_out_split(interactions)
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")

    model = FastALSModel(
        interactions=train_interactions,
        config=config,
    )

    print(f"Users: {model.user_count}")
    print(f"Items: {model.item_count}")
    print(f"U shape: {model.U.shape}")
    print(f"V shape: {model.V.shape}")
    print(f"SU shape: {model.SU.shape}")
    print(f"SV shape: {model.SV.shape}")
    print(f"Wi min/max: {model.Wi.min():.8f} / {model.Wi.max():.8f}")

    train_model(model)

    hr = hit_rate_at_k(model, test_interactions, k=config.top_k)
    print(f"Hit Rate@{config.top_k}: {hr:.4f}")

    sample_users = [u for u, _ in test_interactions[:3]]
    for raw_user_id in sample_users:
        recs = recommend_top_k(model, raw_user_id, k=5)
        print(f"Top-5 recommendations for user {raw_user_id}: {recs}")

    spark.stop()


if __name__ == "__main__":
    main()