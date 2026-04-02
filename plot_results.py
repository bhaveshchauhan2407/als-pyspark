import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def save_line_chart(df, x_col, y_cols, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    for y_col in y_cols:
        plt.plot(df[x_col], df[y_col], marker="o", label=y_col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_runtime_chart(df, x_col, title, xlabel, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df["runtime_seconds"], marker="o", label="runtime_seconds")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_k_charts():
    off = load_csv(os.path.join(RESULTS_DIR, "k_sweep_offline.csv"))
    on = load_csv(os.path.join(RESULTS_DIR, "k_sweep_online.csv"))
    if off is None or on is None:
        return

    save_line_chart(
        off, "factors", ["hr_at_k", "ndcg_at_k"],
        "Offline K Sweep: HR@K and NDCG@K",
        "Number of latent factors (K)",
        "Metric value",
        os.path.join(FIGURES_DIR, "k_sweep_offline_metrics.png")
    )

    save_line_chart(
        on, "factors", ["hr_at_k", "ndcg_at_k"],
        "Online K Sweep: HR@K and NDCG@K",
        "Number of latent factors (K)",
        "Metric value",
        os.path.join(FIGURES_DIR, "k_sweep_online_metrics.png")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(off["factors"], off["runtime_seconds"], marker="o", label="offline")
    plt.plot(on["factors"], on["runtime_seconds"], marker="o", label="online")
    plt.title("K Sweep Runtime")
    plt.xlabel("Number of latent factors (K)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "k_sweep_runtime.png"), dpi=300)
    plt.close()


def make_w0_charts():
    off = load_csv(os.path.join(RESULTS_DIR, "w0_sweep_offline.csv"))
    on = load_csv(os.path.join(RESULTS_DIR, "w0_sweep_online.csv"))
    if off is None or on is None:
        return

    save_line_chart(
        off, "w0", ["hr_at_k", "ndcg_at_k"],
        "Offline w0 Sweep: HR@K and NDCG@K",
        "w0",
        "Metric value",
        os.path.join(FIGURES_DIR, "w0_sweep_offline_metrics.png")
    )

    save_line_chart(
        on, "w0", ["hr_at_k", "ndcg_at_k"],
        "Online w0 Sweep: HR@K and NDCG@K",
        "w0",
        "Metric value",
        os.path.join(FIGURES_DIR, "w0_sweep_online_metrics.png")
    )


def make_alpha_charts():
    off = load_csv(os.path.join(RESULTS_DIR, "alpha_sweep_offline.csv"))
    on = load_csv(os.path.join(RESULTS_DIR, "alpha_sweep_online.csv"))
    if off is None or on is None:
        return

    save_line_chart(
        off, "alpha", ["hr_at_k", "ndcg_at_k"],
        "Offline Alpha Sweep: HR@K and NDCG@K",
        "alpha",
        "Metric value",
        os.path.join(FIGURES_DIR, "alpha_sweep_offline_metrics.png")
    )

    save_line_chart(
        on, "alpha", ["hr_at_k", "ndcg_at_k"],
        "Online Alpha Sweep: HR@K and NDCG@K",
        "alpha",
        "Metric value",
        os.path.join(FIGURES_DIR, "alpha_sweep_online_metrics.png")
    )


def make_iteration_charts():
    off = load_csv(os.path.join(RESULTS_DIR, "iteration_sweep_offline.csv"))
    on = load_csv(os.path.join(RESULTS_DIR, "iteration_sweep_online.csv"))
    if off is None or on is None:
        return

    save_line_chart(
        off, "max_iter", ["hr_at_k", "ndcg_at_k"],
        "Offline Iteration Sweep: HR@K and NDCG@K",
        "max_iter",
        "Metric value",
        os.path.join(FIGURES_DIR, "iteration_sweep_offline_metrics.png")
    )

    save_line_chart(
        on, "max_iter", ["hr_at_k", "ndcg_at_k"],
        "Online Iteration Sweep: HR@K and NDCG@K",
        "max_iter",
        "Metric value",
        os.path.join(FIGURES_DIR, "iteration_sweep_online_metrics.png")
    )

    save_line_chart(
        off, "max_iter", ["final_loss"],
        "Offline Iteration Sweep: Final Loss",
        "max_iter",
        "Final loss",
        os.path.join(FIGURES_DIR, "iteration_sweep_loss.png")
    )


def make_scalability_charts():
    off = load_csv(os.path.join(RESULTS_DIR, "scalability_sweep_offline.csv"))
    on = load_csv(os.path.join(RESULTS_DIR, "scalability_sweep_online.csv"))
    if off is None or on is None:
        return

    save_runtime_chart(
        off, "row_limit",
        "Offline Scalability: Runtime vs Data Size",
        "Row limit",
        os.path.join(FIGURES_DIR, "scalability_offline_runtime.png")
    )

    save_runtime_chart(
        on, "row_limit",
        "Online Scalability: Runtime vs Data Size",
        "Row limit",
        os.path.join(FIGURES_DIR, "scalability_online_runtime.png")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(off["row_limit"], off["hr_at_k"], marker="o", label="offline_hr")
    plt.plot(off["row_limit"], off["ndcg_at_k"], marker="o", label="offline_ndcg")
    plt.plot(on["row_limit"], on["hr_at_k"], marker="o", label="online_hr")
    plt.plot(on["row_limit"], on["ndcg_at_k"], marker="o", label="online_ndcg")
    plt.title("Scalability: Ranking Metrics vs Data Size")
    plt.xlabel("Row limit")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "scalability_metrics.png"), dpi=300)
    plt.close()


def main():
    ensure_figures_dir()
    make_k_charts()
    make_w0_charts()
    make_alpha_charts()
    make_iteration_charts()
    make_scalability_charts()
    print("Finished generating charts in figures/")


if __name__ == "__main__":
    main()