import os
import time
import itertools
import pandas as pd

from src.config import FastALSConfig
from src.model import FastALSModel
from src.train import train_model
from src.evaluate import hit_rate_at_k, ndcg_at_k
from src.split import leave_one_out_split


def ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)


def run_single_experiment(
    interactions,
    factors=8,
    max_iter=3,
    reg=0.1,
    w0=1.0,
    alpha=0.5,
    top_k=10,
    init_mean=0.0,
    init_stdev=0.001,
    random_seed=42,
    show_progress=False,
    show_loss=False,
):
    train_interactions, test_interactions = leave_one_out_split(interactions)

    config = FastALSConfig(
        factors=factors,
        max_iter=max_iter,
        reg=reg,
        w0=w0,
        alpha=alpha,
        init_mean=init_mean,
        init_stdev=init_stdev,
        show_progress=show_progress,
        show_loss=show_loss,
        top_k=top_k,
        random_seed=random_seed,
    )

    start_time = time.time()

    model = FastALSModel(
        interactions=train_interactions,
        config=config,
    )

    train_model(model)

    elapsed = time.time() - start_time
    final_loss = model.loss()
    hr = hit_rate_at_k(model, test_interactions, k=top_k)
    ndcg = ndcg_at_k(model, test_interactions, k=top_k)

    result = {
        "factors": factors,
        "max_iter": max_iter,
        "reg": reg,
        "w0": w0,
        "alpha": alpha,
        "top_k": top_k,
        "train_interactions": len(train_interactions),
        "test_interactions": len(test_interactions),
        "user_count": model.user_count,
        "item_count": model.item_count,
        "final_loss": final_loss,
        "hr_at_k": hr,
        "ndcg_at_k": ndcg,
        "runtime_seconds": elapsed,
    }

    return result


def run_grid_experiments(
    interactions,
    factors_list,
    max_iter_list,
    reg_list,
    w0_list,
    alpha_list,
    top_k=10,
    output_csv="results/experiment_results.csv",
):
    ensure_results_dir("results")

    all_results = []

    combinations = list(
        itertools.product(
            factors_list,
            max_iter_list,
            reg_list,
            w0_list,
            alpha_list,
        )
    )

    print(f"Total experiment combinations: {len(combinations)}")

    for idx, (factors, max_iter, reg, w0, alpha) in enumerate(combinations, start=1):
        print(
            f"\nRunning experiment {idx}/{len(combinations)} | "
            f"K={factors}, iter={max_iter}, reg={reg}, w0={w0}, alpha={alpha}"
        )

        result = run_single_experiment(
            interactions=interactions,
            factors=factors,
            max_iter=max_iter,
            reg=reg,
            w0=w0,
            alpha=alpha,
            top_k=top_k,
            show_progress=False,
            show_loss=False,
        )

        all_results.append(result)

        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)

    return pd.DataFrame(all_results)