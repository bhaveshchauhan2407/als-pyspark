import os
import time
import pandas as pd

from src.config import FastALSConfig
from src.model import FastALSModel
from src.train import train_model
from src.evaluate import hit_rate_at_k, ndcg_at_k, online_protocol_metrics
from src.split import leave_one_out_split
from src.online_split import chronological_90_10_split


def ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)


def build_model(
    train_interactions,
    factors=8,
    max_iter=5,
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

    model = FastALSModel(
        interactions=train_interactions,
        config=config,
    )
    return model


def run_offline_experiment(interactions, **params):
    train_interactions, test_interactions = leave_one_out_split(interactions)

    start = time.time()
    model = build_model(train_interactions, **params)
    train_model(model)
    runtime_seconds = time.time() - start

    return {
        "protocol": "offline",
        "factors": params["factors"],
        "max_iter": params["max_iter"],
        "reg": params["reg"],
        "w0": params["w0"],
        "alpha": params["alpha"],
        "top_k": params["top_k"],
        "train_interactions": len(train_interactions),
        "test_interactions": len(test_interactions),
        "user_count": model.user_count,
        "item_count": model.item_count,
        "final_loss": model.loss(),
        "hr_at_k": hit_rate_at_k(model, test_interactions, k=params["top_k"]),
        "ndcg_at_k": ndcg_at_k(model, test_interactions, k=params["top_k"]),
        "runtime_seconds": runtime_seconds,
    }


def run_online_experiment(events, w_new=4.0, online_iter=1, **params):
    train_interactions, test_interactions = chronological_90_10_split(events)

    start = time.time()
    model = build_model(train_interactions, **params)
    train_model(model)

    metrics = online_protocol_metrics(
        model,
        test_interactions,
        k=params["top_k"],
        w_new=w_new,
        online_iter=online_iter,
    )
    runtime_seconds = time.time() - start

    return {
        "protocol": "online",
        "factors": params["factors"],
        "max_iter": params["max_iter"],
        "reg": params["reg"],
        "w0": params["w0"],
        "alpha": params["alpha"],
        "top_k": params["top_k"],
        "w_new": w_new,
        "online_iter": online_iter,
        "train_interactions": len(train_interactions),
        "test_interactions": len(test_interactions),
        "user_count": model.user_count,
        "item_count": model.item_count,
        "final_loss": model.loss(),
        "hr_at_k": metrics["hr_at_k"],
        "ndcg_at_k": metrics["ndcg_at_k"],
        "runtime_seconds": runtime_seconds,
    }


def run_single_parameter_sweep_both_protocols(
    interactions,
    events,
    sweep_name,
    sweep_values,
    fixed_params,
    offline_output_csv,
    online_output_csv,
    w_new=4.0,
    online_iter=1,
):
    ensure_results_dir("results")

    offline_results = []
    online_results = []

    for idx, value in enumerate(sweep_values, start=1):
        params = fixed_params.copy()
        params[sweep_name] = value

        print(f"Running {sweep_name} sweep {idx}/{len(sweep_values)} | {sweep_name}={value}")

        offline_result = run_offline_experiment(interactions=interactions, **params)
        online_result = run_online_experiment(
            events=events,
            w_new=w_new,
            online_iter=online_iter,
            **params,
        )

        offline_results.append(offline_result)
        online_results.append(online_result)

        pd.DataFrame(offline_results).to_csv(offline_output_csv, index=False)
        pd.DataFrame(online_results).to_csv(online_output_csv, index=False)

    return pd.DataFrame(offline_results), pd.DataFrame(online_results)