from src.spark_loader import (
    create_spark_session,
    load_yelp_interactions_spark,
    load_yelp_events_spark,
    spark_df_to_events,
)
from src.rdd_loader import (
    spark_df_to_rdd,
    collect_rdd_interactions,
)
from src.sweeps import run_single_parameter_sweep_both_protocols

BASELINE_PARAMS = {
    "factors": 12,
    "max_iter": 5,
    "reg": 0.1,
    "w0": 1.0,
    "alpha": 0.5,
    "top_k": 10,
    "init_mean": 0.0,
    "init_stdev": 0.001,
    "random_seed": 42,
}


def main():
    spark = create_spark_session()

    offline_df = load_yelp_interactions_spark(spark, "data/yelp.rating", limit_rows=10000)
    interactions = collect_rdd_interactions(spark_df_to_rdd(offline_df))

    online_df = load_yelp_events_spark(spark, "data/yelp.rating", limit_rows=10000)
    events = spark_df_to_events(online_df)

    offline_df_results, online_df_results = run_single_parameter_sweep_both_protocols(
        interactions=interactions,
        events=events,
        sweep_name="factors",
        sweep_values=[4, 8, 12, 16, 24, 32],
        fixed_params=BASELINE_PARAMS,
        offline_output_csv="results/k_sweep_offline.csv",
        online_output_csv="results/k_sweep_online.csv",
        w_new=4.0,
        online_iter=1,
    )

    print("\nOFFLINE RESULTS")
    print(offline_df_results)

    print("\nONLINE RESULTS")
    print(online_df_results)

    spark.stop()


if __name__ == "__main__":
    main()