from src.spark_loader import create_spark_session, load_yelp_interactions_spark
from src.rdd_loader import spark_df_to_rdd, collect_rdd_interactions
from src.experiments import run_grid_experiments


def main():
    spark = create_spark_session()

    data_path = "data/yelp.rating"

    df = load_yelp_interactions_spark(spark, data_path, limit_rows=10000)

    interactions_rdd = spark_df_to_rdd(df)
    interactions = collect_rdd_interactions(interactions_rdd)

    results_df = run_grid_experiments(
        interactions=interactions,
        factors_list=[4, 8, 12, 16],
        max_iter_list=[3, 5],
        reg_list=[0.1],
        w0_list=[0.5, 1.0, 2.0],
        alpha_list=[0.0, 0.25, 0.5, 0.75],
        top_k=10,
        output_csv="results/experiment_results.csv",
    )

    print("\nFinished experiments.")
    print(results_df.sort_values(by="hr_at_k", ascending=False).head(10))

    spark.stop()


if __name__ == "__main__":
    main()