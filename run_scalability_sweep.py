import pandas as pd

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
from src.sweeps import ensure_results_dir, run_offline_experiment, run_online_experiment


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
    ensure_results_dir("results")

    row_sizes = [10000, 20000, 50000, 100000]

    offline_results = []
    online_results = []

    for idx, row_limit in enumerate(row_sizes, start=1):
        print(f"Running scalability sweep {idx}/{len(row_sizes)} | rows={row_limit}")

        offline_df = load_yelp_interactions_spark(spark, "data/yelp.rating", limit_rows=row_limit)
        interactions = collect_rdd_interactions(spark_df_to_rdd(offline_df))

        online_df = load_yelp_events_spark(spark, "data/yelp.rating", limit_rows=row_limit)
        events = spark_df_to_events(online_df)

        offline_result = run_offline_experiment(interactions=interactions, **BASELINE_PARAMS)
        offline_result["row_limit"] = row_limit
        offline_results.append(offline_result)

        online_result = run_online_experiment(
            events=events,
            w_new=4.0,
            online_iter=1,
            **BASELINE_PARAMS,
        )
        online_result["row_limit"] = row_limit
        online_results.append(online_result)

        pd.DataFrame(offline_results).to_csv("results/scalability_sweep_offline.csv", index=False)
        pd.DataFrame(online_results).to_csv("results/scalability_sweep_online.csv", index=False)

    print("\nOFFLINE RESULTS")
    print(pd.DataFrame(offline_results))

    print("\nONLINE RESULTS")
    print(pd.DataFrame(online_results))

    spark.stop()


if __name__ == "__main__":
    main()