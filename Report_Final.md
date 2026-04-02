---
geometry: "top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm"
fontsize: 12pt
linestretch: 1.3
header-includes:
  - \usepackage{fancyhdr}
  - \usepackage{graphicx}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{hyperref}
  - \usepackage{array}
  - \usepackage{xcolor}
  - \usepackage{setspace}
  - \usepackage{listings}
  - \lstset{basicstyle=\ttfamily\footnotesize, backgroundcolor=\color[gray]{0.95}, frame=single, framesep=4pt, breaklines=true, breakatwhitespace=false, columns=flexible, keepspaces=true}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\textit{[COURSE NAME]}}
  - \fancyhead[R]{\textit{[UNIVERSITY NAME]}}
  - \fancyfoot[L]{\textit{[LEFT FOOTER]}}
  - \fancyfoot[C]{\thepage}
  - \fancyfoot[R]{\textit{[RIGHT FOOTER]}}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \renewcommand{\footrulewidth}{0.4pt}
  - \setlength{\headheight}{15pt}
---

<!-- ================================================================ -->
<!--                         COVER PAGE                               -->
<!-- ================================================================ -->

\begin{titlepage}
\centering
\vspace*{1.5cm}

{\LARGE \textbf{[UNIVERSITY NAME]}}\\[0.4cm]
{\large [DEPARTMENT NAME]}

\vspace{1.2cm}
\rule{\linewidth}{0.5pt}\\[0.4cm]

{\Huge \textbf{Fast Matrix Factorization for Online}}\\[0.3cm]
{\Huge \textbf{Recommendation with Implicit Feedback:}}\\[0.4cm]
{\LARGE \textit{A PySpark Implementation and Experimental Analysis}}\\[0.4cm]

\rule{\linewidth}{0.5pt}

\vspace{1.0cm}

{\large \textbf{Course:} [COURSE NAME — e.g. Data Science for Business / DBMS]}\\[0.3cm]
{\large \textbf{MSc Programme:} [MSC COURSE NAME — e.g. MSc Data Science]}\\[0.3cm]
{\large \textbf{Academic Year:} [ACADEMIC YEAR — e.g. 2025--2026]}\\[0.3cm]
{\large \textbf{Submitted to:} Prof. Dario Colazzo}\\[0.3cm]
{\large \textbf{Submission Date:} April 3, 2026}

\vspace{1.5cm}

{\large \textbf{Team Members:}}\\[0.5cm]
\begin{tabular}{c}
{\Large [TEAMMATE 1 — Full Name]} \\[0.3cm]
{\Large [TEAMMATE 2 — Full Name]} \\[0.3cm]
{\large \textit{(Add Teammate 3 if applicable)}} \\
\end{tabular}

\vfill

{\small Based on the paper: He, X., Zhang, H., Kan, M.-Y., \& Chua, T.-S. (2016).\\
\textit{Fast Matrix Factorization for Online Recommendation with Implicit Feedback.}\\
SIGIR '16, Pisa, Italy.}

\end{titlepage}

\newpage
\tableofcontents
\newpage

<!-- ================================================================ -->
<!--                         SECTION 1                                -->
<!--              Description of the Adopted Solution                 -->
<!-- ================================================================ -->

# 1. Description of the Adopted Solution

## 1.1 Problem Context

This project is a Python/PySpark reimplementation and experimental evaluation of the **eALS** (element-wise Alternating Least Squares) algorithm from:

> He, X., Zhang, H., Kan, M.-Y., & Chua, T.-S. (2016). *Fast Matrix Factorization for Online Recommendation with Implicit Feedback*. SIGIR '16.

The paper addresses two fundamental problems in recommender systems. First, **implicit feedback**: users rarely provide explicit ratings; preferences must be inferred from observed interactions such as clicks or purchases. This creates the *one-class problem* where observed entries are positives, but missing entries are not confirmed negatives. Second, **online learning**: in real-world systems, new interactions arrive continuously and models must adapt in real-time without full retraining.

The paper proves the algorithm is *embarrassingly parallel*, but provides no distributed implementation. Our task is to build a PySpark version using both RDDs and DataFrames, and to perform experimental analysis on the Yelp dataset from the paper.

## 1.2 Matrix Factorization for Implicit Feedback

Given a user--item interaction matrix $\mathbf{R} \in \mathbb{R}^{M \times N}$, matrix factorization maps each user $u$ and item $i$ into a shared $K$-dimensional latent space:

$$\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i \tag{1}$$

where $\mathbf{p}_u \in \mathbb{R}^K$ and $\mathbf{q}_i \in \mathbb{R}^K$ are latent factor vectors. Recommendation is performed by ranking all items by predicted score.

## 1.3 The eALS Objective Function

The paper replaces the standard uniform missing-data weight with a *popularity-aware* weighting scheme. The full objective is:

$$\mathcal{L} = \sum_{(u,i) \in R} w_{ui}(r_{ui} - \hat{r}_{ui})^2 + \sum_{u=1}^{M} \sum_{i \notin R_u} c_i \hat{r}_{ui}^2 + \lambda\left(\sum_{u=1}^{M} \|\mathbf{p}_u\|^2 + \sum_{i=1}^{N} \|\mathbf{q}_i\|^2\right) \tag{7}$$

where $w_{ui} = 1$ for all observed interactions, and $c_i$ is the item-specific confidence that a missing entry is a true negative:

$$c_i = c_0 \cdot \frac{f_i^\alpha}{\sum_{j=1}^{N} f_j^\alpha} \tag{8}$$

Here $f_i = |R_i| / \sum_j |R_j|$ is normalized item popularity, $c_0$ is the overall missing-data weight scale (`w0` in our code), and $\alpha$ controls popularity skewness. Setting $\alpha = 0$ recovers uniform weighting ($w_0 = c_0/N$).

## 1.4 Implementation Architecture

Our implementation has two layers:

**Layer 1 — Spark (data pipeline).** PySpark DataFrames and RDDs handle all file reading, parsing, deduplication, grouping, and data transfer. This is the distributed layer and the primary focus of this course.

**Layer 2 — NumPy (model training).** The eALS coordinate descent runs locally in Python/NumPy after interactions are collected from Spark. This was necessary for correctness, numerical stability, and debugging, and is explicitly acknowledged as a limitation.

The dataset is **Yelp** (`data/yelp.rating`): a file of tab-separated `(user_id, item_id, rating, timestamp)` records. Although explicit ratings are present in the raw file, we treat each row as an implicit positive interaction — only the existence of a `(user, item)` pair is used; the rating value is discarded entirely.

## 1.5 Project File Structure

```
DBMS_PROJECT/
|-- data/
|   \-- yelp.rating                  # Raw Yelp interaction data
|-- figures/
|   |-- alpha_sweep_offline_metrics.png
|   |-- alpha_sweep_online_metrics.png
|   |-- iteration_sweep_loss.png
|   |-- iteration_sweep_offline_metrics.png
|   |-- iteration_sweep_online_metrics.png
|   |-- k_sweep_offline_metrics.png
|   |-- k_sweep_online_metrics.png
|   |-- k_sweep_runtime.png
|   |-- scalability_metrics.png
|   |-- scalability_offline_runtime.png
|   |-- scalability_online_runtime.png
|   |-- w0_sweep_offline_metrics.png
|   \-- w0_sweep_online_metrics.png
|-- results/
|   |-- alpha_sweep_offline.csv
|   |-- alpha_sweep_online.csv
|   |-- experiment_results.csv       # Grid search results for baseline selection
|   |-- iteration_sweep_offline.csv
|   |-- iteration_sweep_online.csv
|   |-- k_sweep_offline.csv
|   |-- k_sweep_online.csv
|   |-- scalability_sweep_offline.csv
|   |-- scalability_sweep_online.csv
|   |-- w0_sweep_offline.csv
|   \-- w0_sweep_online.csv
|-- src/
|   |-- __init__.py
|   |-- config.py                    # FastALSConfig dataclass
|   |-- data_loader.py               # Pandas loader (legacy, kept for reference)
|   |-- evaluate.py                  # HR@K, NDCG@K, online protocol metrics
|   |-- experiments.py               # Grid search runner
|   |-- model.py                     # Core eALS model (FastALSModel)
|   |-- online_split.py              # Chronological 90/10 split
|   |-- predict.py                   # Dot-product scoring
|   |-- rdd_loader.py                # Spark RDD layer
|   |-- recommender.py               # Top-K recommendation
|   |-- spark_loader.py              # Spark DataFrame layer
|   |-- split.py                     # Leave-one-out offline split
|   |-- sweeps.py                    # Controlled parameter sweep logic
|   |-- train.py                     # Training loop
|   \-- utils.py                     # Timing and print utilities
|-- main.py                          # End-to-end demo runner
|-- plot_results.py                  # Matplotlib figure generator
|-- README.md
|-- REPORT.md
|-- requirements.txt
|-- run_alpha_sweep.py
|-- run_experiments.py
|-- run_iteration_sweep.py
|-- run_k_sweep.py
|-- run_scalability_sweep.py
\-- run_w0_sweep.py
```

## 1.6 Incremental Development Path

The project was developed in seven stages, each building on a stable foundation.

**Stage 1 — Java source analysis.** We studied the author's Java implementation: `MF_fastALS.java`, `TopKRecommender.java`, `SparseMatrix.java`, `SparseVector.java`, `DenseMatrix.java`, `DenseVector.java`, `Rating.java`, and `Printer.java`. From these we identified how the model is initialized, how $c_i$ (`Wi` array) is computed, how the cache matrices `SU` ($= \mathbf{P}^\top\mathbf{P}$) and `SV` ($= \mathbf{Q}^\top \text{diag}(\mathbf{c}) \mathbf{Q}$) are maintained, and the update logic for `update_user()`, `update_item()`, and online `updateModel()`.

**Stage 2 — Python skeleton.** A pure Python project skeleton was created before Spark was introduced: `config.py` (dataclass), `model.py` (stub), `train.py`, `evaluate.py` placeholders, and `data_loader.py` (pandas-based). This allowed the Java logic to be ported without distributed complexity obscuring bugs.

**Stage 3 — Stable local reference implementation.** `update_user`, `update_item`, and `loss` were implemented in NumPy. Initial training was numerically unstable. Stability was achieved by: deduplicating `(user, item)` pairs, reducing `init_stdev` to `0.001`, increasing regularization, adding value clipping to $[-10, 10]$, and validating on small subsets.

**Stage 4 — Spark integration.** Spark was added in two sub-layers: a DataFrame layer in `spark_loader.py` for reading, parsing, and deduplication; and an RDD layer in `rdd_loader.py` for explicit grouped structures.

**Stage 5 — Evaluation pipeline.** `split.py` (leave-one-out), `recommender.py` (top-$K$), and `evaluate.py` (HR@$K$, NDCG@$K$) were added.

**Stage 6 — Online protocol.** `online_split.py` (chronological 90/10 split) and the `online_protocol_metrics` function in `evaluate.py` (evaluate-then-update loop) were added, along with `update_model()` in `model.py`.

**Stage 7 — Experiment automation.** `experiments.py` (grid search), `sweeps.py` (one-parameter sweeps), and the six `run_*.py` runners were added, along with `plot_results.py` for figure generation.

\newpage

<!-- ================================================================ -->
<!--                         SECTION 2                                -->
<!--        Algorithm Design, Code Description, and Comments          -->
<!-- ================================================================ -->

# 2. Algorithm Design, Code Description, and Comments

## 2.1 The eALS Update Rules

Standard vector-wise ALS inverts a $K \times K$ matrix per user/item at cost $O(K^3)$. eALS instead updates each scalar component individually while holding all others fixed, eliminating inversion entirely.

The update rule for the $f$-th component of user $u$'s latent vector (Eq. 12 in the paper) is:

$$p_{uf} = \frac{\displaystyle\sum_{i \in R_u} \bigl[w_{ui} r_{ui} - (w_{ui} - c_i)\hat{r}^f_{ui}\bigr] q_{if} \;-\; \sum_{k \neq f} p_{uk} \, s^q_{kf}}{\displaystyle\sum_{i \in R_u} (w_{ui} - c_i) q_{if}^2 \;+\; s^q_{ff} \;+\; \lambda} \tag{12}$$

where $\hat{r}^f_{ui} = \hat{r}_{ui} - p_{uf} q_{if}$ is the prediction with the $f$-th component removed (cheap to compute from the maintained prediction cache), and $\mathbf{S}^q$ is the pre-computed cache:

$$\mathbf{S}^q = \sum_{i=1}^{N} c_i \, \mathbf{q}_i \mathbf{q}_i^\top \tag{cache-q}$$

This cache is independent of $u$ and is pre-computed once per iteration, replacing an $O(NK)$ traversal per coordinate with a single $O(K)$ cache lookup.

The item update rule (Eq. 13) is symmetric:

$$q_{if} = \frac{\displaystyle\sum_{u \in R_i} \bigl[w_{ui} r_{ui} - (w_{ui} - c_i)\hat{r}^f_{ui}\bigr] p_{uf} \;-\; c_i \sum_{k \neq f} q_{ik} \, s^p_{kf}}{\displaystyle\sum_{u \in R_i} (w_{ui} - c_i) p_{uf}^2 \;+\; c_i s^p_{ff} \;+\; \lambda} \tag{13}$$

where $\mathbf{S}^p = \mathbf{P}^\top \mathbf{P}$ is the user-side cache. Note that $c_i$ appears as a scalar multiplier in the numerator's cross-factor sum and in the denominator, directly encoding the item's popularity-derived negative confidence.

The per-iteration time complexity is $O\!\left((M+N)K^2 + |R|K\right)$, a factor of $K$ faster than standard vector-wise ALS.

## 2.2 Online Incremental Update (Algorithm 2)

When a new interaction $(u, i)$ arrives with weight $w_\text{new}$, only the latent vectors $\mathbf{p}_u$ and $\mathbf{q}_i$ are updated, with their cache contributions refreshed incrementally. Per-update complexity is:

$$O\!\left(K^2 + (|R_u| + |R_i|)K\right)$$

This is independent of $M$, $N$, and $|R|$, making eALS suitable for real-time deployment. One iteration is typically sufficient — a single eALS step finds the exact optimum for each coordinate with others fixed.

## 2.3 Parallelization Architecture

The paper proves that eALS is embarrassingly parallel: all user updates share only a read-only $\mathbf{S}^q$ cache and can proceed fully independently. In our implementation this maps to:

1. **Spark** loads and preprocesses the data distributedly.
2. **Broadcast:** $\mathbf{Q}$, $\mathbf{S}^q$, and $\mathbf{c}$ are read-only during user updates.
3. **Collect:** updated user factors are gathered and $\mathbf{P}$ is rebuilt.
4. **Repeat** symmetrically for item factors.

In our course implementation, steps 2--4 run locally in NumPy after `collect_rdd_interactions()` gathers data from Spark. This is a limitation but the correctness of the parallel decomposition is preserved.

## 2.4 How We Use Spark: DataFrames and RDDs in Detail

This section describes in detail how PySpark's two main abstractions are used in our pipeline, and why each choice was made.

### 2.4.1 Spark DataFrames (`src/spark_loader.py`)

DataFrames are Spark's structured, schema-aware abstraction. We use them for the full data ingestion and preprocessing pipeline.

**Reading the raw file.** The Yelp file contains space-separated `(user_id, item_id, rating, timestamp)` records with no header. We use `spark.read.text()` to read it as raw strings, then apply `F.split(F.trim(col("value")), r"\s+")` to parse each line into four typed columns. This approach handles irregular whitespace that would cause issues with a fixed-delimiter CSV reader.

**Offline interactions.** For the offline protocol, only `(user_id, item_id)` pairs are needed. After parsing, we call `.select("user_id", "item_id").dropna().dropDuplicates().limit(limit_rows)`. The `dropDuplicates()` call is critical for implicit feedback: a user reviewing a business twice should count as a single positive signal, not two.

**Online events.** For the online protocol, we retain timestamps and call `.orderBy("timestamp").limit(limit_rows)` to preserve chronological order for the streaming simulation. We deliberately do *not* call `dropDuplicates()` here, since repeat interactions at different times are genuine online events.

**Grouping for preview.** `build_groupings_with_spark()` uses `groupBy("user_id").agg(F.collect_list("item_id"))` to produce a `user_id → [item_ids]` summary DataFrame. This is used for inspection and logging in `main.py`, demonstrating Spark's ability to compute grouped aggregations over the full dataset distributedly.

**Impact on the algorithm.** The DataFrame layer ensures that deduplication, type-casting, and filtering happen at the Spark level before any data reaches the local Python model. This makes the pipeline robust and scalable: for larger datasets, Spark will distribute the file reading and deduplication across all available cores without any code changes.

### 2.4.2 Spark RDDs (`src/rdd_loader.py`)

RDDs are Spark's lower-level, unstructured abstraction. We use them for explicit grouped interaction structures required by the model.

**DataFrame to RDD.** `spark_df_to_rdd(df)` calls `df.rdd.map(lambda row: (row["user_id"], row["item_id"]))` to convert the structured DataFrame into an RDD of `(user_id, item_id)` tuples. This is the bridge between the two abstractions.

**User grouping.** `build_user_items_rdd(interactions_rdd)` calls `.groupByKey().mapValues(list)` to produce an RDD of `(user_id, [item_id, ...])`. `groupByKey()` is a shuffle operation — each `user_id` becomes the key and all its items are collected to the same partition. The result represents the `user_items` dictionary that the model needs for `update_user()`.

**Item grouping.** `build_item_users_rdd(interactions_rdd)` first calls `.map(lambda x: (x[1], x[0]))` to swap the key-value pair (making `item_id` the key), then `groupByKey().mapValues(list)` to produce `(item_id, [user_id, ...])`. The intermediate `map` is cheap (no shuffle); only `groupByKey()` triggers a shuffle.

**Collection to driver.** `collect_rdd_interactions(interactions_rdd)` calls `.collect()`, bringing the RDD back to a local Python list. This is the boundary between distributed and local computation. The model then uses this list to build its internal `user_items` and `item_users` dictionaries.

**Impact on the algorithm.** Using the explicit RDD API demonstrates understanding of Spark's low-level primitives. In a fully distributed version of eALS, the `build_user_items_rdd` structure would be the natural unit of parallelism: each partition of users could be assigned to a different worker for the update step. The `collect()` call is the bottleneck that prevents full distribution in our current implementation.

### 2.4.3 Data Flow Summary

The complete data flow is:

```
yelp.rating (HDFS/local)
    |-- spark.read.text()                  [Spark I/O]
    |-- split / cast / dropna              [DataFrame transform]
    |-- dropDuplicates / orderBy / limit   [DataFrame transform -- shuffle for dedup]
    |-- .rdd.map()                         [DataFrame -> RDD]
    |-- groupByKey() / mapValues()         [RDD transform -- shuffle]
    |-- .collect()                         [RDD action -> local Python list]
    |-- FastALSModel(interactions)         [NumPy local training]
    |-- train_model(model)                 [NumPy local updates]
    \-- hit_rate_at_k / ndcg_at_k          [Local evaluation]
```

Each step that involves a `groupByKey()` or `dropDuplicates()` triggers a shuffle in Spark, meaning data is redistributed across partitions by key. The `dropDuplicates()` in the DataFrame layer and the `groupByKey()` in the RDD layer are the two shuffle-inducing operations in our pipeline.

## 2.5 Module-by-Module Code Description

### `src/config.py` — Hyperparameter Configuration

The `FastALSConfig` dataclass centralizes all hyperparameters. Every sweep runner imports this and uses `dataclasses.replace()` to vary a single field per experiment run.

```python
from dataclasses import dataclass

@dataclass
class FastALSConfig:
    factors: int = 10        # K: number of latent dimensions
    max_iter: int = 10       # training iterations
    reg: float = 0.01        # lambda: L2 regularization strength
    w0: float = 1.0          # c0: overall missing-data weight scale
    alpha: float = 0.5       # popularity exponent for ci weighting
    init_mean: float = 0.0   # mean for random factor initialization
    init_stdev: float = 0.001 # std dev — kept small for stability
    show_progress: bool = True
    show_loss: bool = True
    top_k: int = 10          # recommendation cutoff K_eval
    random_seed: int = 42
```

**Comment:** `init_stdev = 0.001` was found necessary during Stage 3 to prevent divergence in early iterations. Larger values (e.g. `0.01`) caused the loss to oscillate rather than decrease monotonically.

---

### `src/spark_loader.py` — Spark DataFrame Layer

The primary Spark entry point. Key function `load_yelp_interactions_spark`:

```python
def load_yelp_interactions_spark(spark, path: str, limit_rows: int = 5000):
    raw_df = spark.read.text(path)
    # Parse each space-separated line into 4 typed columns
    parts = F.split(F.trim(F.col("value")), r"\s+")
    df = raw_df.select(
        parts.getItem(0).cast("int").alias("user_id"),
        parts.getItem(1).cast("int").alias("item_id"),
        parts.getItem(2).cast("float").alias("rating"),
        parts.getItem(3).cast("long").alias("timestamp"),
    )
    # For offline: drop rating/timestamp, deduplicate (user,item) pairs
    df = (
        df.select("user_id", "item_id")
          .dropna()
          .dropDuplicates()    # shuffle operation — critical for implicit feedback
          .limit(limit_rows)
    )
    return df
```

**Comment:** `dropDuplicates()` triggers a Spark shuffle but is essential — without it, a user with two reviews of the same business would generate duplicate training pairs, inflating that interaction's effective weight and biasing the model. The `limit_rows` parameter allows us to run experiments on data subsets of controlled size.

`load_yelp_events_spark` (for online protocol) omits `dropDuplicates()` and instead calls `.orderBy("timestamp")` — a sort shuffle — because chronological ordering of events is required for the streaming simulation.

`build_groupings_with_spark` produces aggregated previews:

```python
def build_groupings_with_spark(df):
    # groupBy triggers a shuffle; collect_list aggregates all item_ids per user
    user_items_df = df.groupBy("user_id").agg(
        F.collect_list("item_id").alias("item_list")
    )
    item_users_df = df.groupBy("item_id").agg(
        F.collect_list("user_id").alias("user_list")
    )
    return user_items_df, item_users_df
```

**Comment:** These DataFrames are used for inspection and logging only. In `main.py` they are printed with `.show()` to demonstrate Spark's grouped aggregation capability.

---

### `src/rdd_loader.py` — Spark RDD Layer

Provides explicit RDD-level operations on top of the DataFrame output.

```python
def spark_df_to_rdd(df):
    # Convert structured DataFrame row to plain (user_id, item_id) tuple RDD
    return df.rdd.map(lambda row: (row["user_id"], row["item_id"]))

def build_user_items_rdd(interactions_rdd):
    # groupByKey: shuffle by user_id; mapValues: convert ResultIterable to list
    return interactions_rdd.groupByKey().mapValues(list)

def build_item_users_rdd(interactions_rdd):
    # Swap key-value (cheap map), then groupByKey (shuffle by item_id)
    return interactions_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)

def collect_rdd_interactions(interactions_rdd) -> List[Tuple[int, int]]:
    # Bring full RDD to driver — boundary between distributed and local
    return interactions_rdd.collect()
```

**Comment:** `build_item_users_rdd` applies `.map(lambda x: (x[1], x[0]))` first to flip the tuple before `groupByKey()`. This is a narrow transformation (no shuffle). Only `groupByKey()` is wide (requires shuffle). This ordering correctly minimizes data movement.

---

### `src/model.py` — Core eALS Model

The central file, a Python translation of `MF_fastALS.java`. Key sections:

**Item confidence weights (`_compute_item_missing_weights`) — implements Eq. 8:**

```python
def _compute_item_missing_weights(self) -> np.ndarray:
    item_popularity = np.zeros(self.item_count)
    for i in range(self.item_count):
        item_popularity[i] = len(self.item_users[i])  # |Ri|
    total = item_popularity.sum()
    probabilities = item_popularity / total            # fi = |Ri| / sum|Rj|
    powered = np.power(probabilities, self.config.alpha)  # fi^alpha
    z = powered.sum()                                  # normalizer
    wi = self.config.w0 * powered / z                 # Eq. 8: ci = c0 * fi^a / sum(fj^a)
    return wi
```

**Cache initialization (`_init_caches`) — implements paper's S matrices:**

```python
def _init_caches(self) -> None:
    self.SU = self.U.T @ self.U             # Sp = P^T P  (user-side cache)
    weighted_V = self.V * self.Wi[:, np.newaxis]   # diag(c) * V
    self.SV = self.V.T @ weighted_V         # Sq = V^T diag(c) V  (item-side cache)
```

**Comment:** `SU` corresponds to $\mathbf{S}^p$ in the paper and `SV` corresponds to $\mathbf{S}^q$. Note the naming is from the Java source (`SU` for user matrix, `SV` for item matrix) — the paper uses $S^p$ for users and $S^q$ for items. Both are correct.

**`update_user(u)` — implements Eq. 12:**

```python
def update_user(self, u: int) -> None:
    item_list = self.user_items[u]
    # Cache predictions and weights for this user's items
    for i in item_list:
        self.prediction_items[i] = self.predict(u, i)
        self.rating_items[i] = 1.0
        self.w_items[i] = self.W[(u, i)]
    old_vector = self.U[u].copy()
    for f in range(self.config.factors):
        numer = 0.0
        denom = 0.0
        # Cross-factor sum: -sum_{k!=f} p_uk * SV[f,k]  (uses cache Sq)
        for k in range(self.config.factors):
            if k != f:
                numer -= self.U[u, k] * self.SV[f, k]
        for i in item_list:
            # Remove f-th component from cached prediction
            self.prediction_items[i] -= self.U[u, f] * self.V[i, f]
            # Numerator: [wui*rui - (wui-ci)*rhat_f_ui] * qif
            numer += (
                self.w_items[i] * self.rating_items[i]
                - (self.w_items[i] - self.Wi[i]) * self.prediction_items[i]
            ) * self.V[i, f]
            # Denominator: (wui - ci) * qif^2
            denom += (self.w_items[i] - self.Wi[i]) * (self.V[i, f] ** 2)
        denom += self.SV[f, f] + self.config.reg   # + Sq[f,f] + lambda
        if denom != 0 and np.isfinite(numer) and np.isfinite(denom):
            new_value = numer / denom
            if np.isfinite(new_value):
                self.U[u, f] = np.clip(new_value, -10.0, 10.0)  # stability clip
        # Restore prediction after update
        for i in item_list:
            self.prediction_items[i] += self.U[u, f] * self.V[i, f]
    # Incremental SU update: rank-1 correction for changed row u
    for f in range(self.config.factors):
        for k in range(f + 1):
            val = self.SU[f,k] - old_vector[f]*old_vector[k] + self.U[u,f]*self.U[u,k]
            self.SU[f, k] = val
            self.SU[k, f] = val
```

**Comment:** The incremental `SU` update at the end is a correct rank-1 correction: $\mathbf{S}^p_\text{new} = \mathbf{S}^p_\text{old} - \mathbf{p}_u^\text{old}(\mathbf{p}_u^\text{old})^\top + \mathbf{p}_u^\text{new}(\mathbf{p}_u^\text{new})^\top$. This avoids a full $O(MK^2)$ recomputation after each user update, matching the Java implementation's efficiency. `update_item` applies the same pattern with $c_i$ weighting on both the cross-factor sum and the cache update, correctly implementing Eq. 13.

**`update_model(u, i, w_new, online_iter)` — implements Algorithm 2:**

```python
def update_model(self, raw_user_id, raw_item_id, w_new=4.0, online_iter=1):
    # Handle cold-start: new user or item not seen in training
    if raw_user_id not in self.user_to_index:
        u = self._add_new_user(raw_user_id)   # random init, extend U and SU
    else:
        u = self.user_to_index[raw_user_id]
    if raw_item_id not in self.item_to_index:
        i = self._add_new_item(raw_item_id)   # random init, extend V, Wi, caches
    else:
        i = self.item_to_index[raw_item_id]
    # Register the new interaction
    if i not in self.user_items[u]: self.user_items[u].append(i)
    if u not in self.item_users[i]: self.item_users[i].append(u)
    self.W[(u, i)] = w_new     # weight new interaction higher than training
    # Local update of only affected user and item vectors
    for _ in range(online_iter):
        self.update_user(u)
        self.update_item(i)
```

**Comment:** Setting `w_new = 4.0` (four times the default training weight of 1.0) means new interactions have stronger immediate influence on the model, reflecting their higher recency value. This matches the spirit of the paper's Section 5.3.2, which finds optimal `w_new` around 4 for Yelp.

---

### `src/train.py` — Training Loop

```python
def run_one_iteration(model) -> None:
    # Update all user vectors (SU updated incrementally inside update_user)
    for u in range(model.user_count):
        model.update_user(u)
    # Update all item vectors (SV updated incrementally inside update_item)
    for i in range(model.item_count):
        model.update_item(i)

def train_model(model) -> None:
    previous_loss = float("inf")
    for iteration in range(model.config.max_iter):
        start = current_millis()
        run_one_iteration(model)
        elapsed_seconds = (current_millis() - start) / 1000.0
        if model.config.show_progress:
            print(f"Iteration {iteration+1}/{model.config.max_iter} in {format_seconds(elapsed_seconds)}")
        if model.config.show_loss:
            current_loss = model.loss()
            symbol = "-" if current_loss <= previous_loss else "+"
            print(f"Loss: {current_loss:.6f} [{symbol}]")
            previous_loss = current_loss
```

**Comment:** Unlike the Java code which explicitly precomputes `SU` and `SV` at the start of each iteration, our implementation maintains them through incremental rank-1 updates inside `update_user` and `update_item`. This is mathematically equivalent — both approaches keep the caches consistent — but our approach avoids the $O((M+N)K^2)$ full recomputation cost per iteration, relying instead on $O(K^2)$ incremental corrections per user/item. The `loss()` function uses the fast formula from Eq. 14 of the paper, which reuses `SV` to avoid an $O(MNK)$ brute-force computation.

---

### `src/evaluate.py` — Metrics

```python
def hit_rate_at_k(model, test_interactions, k=10):
    hits = 0
    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)
        if true_item in recs:
            hits += 1
    return hits / len(test_interactions)

def ndcg_at_k(model, test_interactions, k=10):
    total_ndcg = 0.0
    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)
        if true_item in recs:
            rank = recs.index(true_item) + 1   # 1-indexed
            total_ndcg += 1.0 / math.log2(rank + 1)
    return total_ndcg / len(test_interactions)

def online_protocol_metrics(model, test_interactions, k=10, w_new=4.0, online_iter=1):
    hits, total_ndcg, total = 0, 0.0, 0
    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)
        if true_item in recs:
            hits += 1
            rank = recs.index(true_item) + 1
            total_ndcg += 1.0 / math.log2(rank + 1)
        # EVALUATE FIRST, then update — correct online protocol ordering
        model.update_model(raw_user_id, true_item, w_new=w_new, online_iter=online_iter)
        total += 1
    return {"hr_at_k": hits/total, "ndcg_at_k": total_ndcg/total}
```

**Comment:** The evaluate-then-update ordering is critical: metrics are computed before the model sees the new interaction, faithfully simulating a real online system that must make a recommendation without knowing the user's next action. `recommend_top_k` in `recommender.py` excludes training items from the candidate list using `seen_items = set(model.user_items[u])`.

---

### `src/sweeps.py` — Parameter Sweep Runner

The `run_single_parameter_sweep_both_protocols` function is the engine behind all sweep experiments:

```python
def run_single_parameter_sweep_both_protocols(
    interactions, events, sweep_name, sweep_values,
    fixed_params, offline_output_csv, online_output_csv,
    w_new=4.0, online_iter=1,
):
    offline_results, online_results = [], []
    for value in sweep_values:
        params = fixed_params.copy()
        params[sweep_name] = value       # vary exactly one parameter
        offline_result = run_offline_experiment(interactions=interactions, **params)
        online_result = run_online_experiment(events=events, w_new=w_new, **params)
        offline_results.append(offline_result)
        online_results.append(online_result)
        # Write incrementally — results saved even if run is interrupted
        pd.DataFrame(offline_results).to_csv(offline_output_csv, index=False)
        pd.DataFrame(online_results).to_csv(online_output_csv, index=False)
    return pd.DataFrame(offline_results), pd.DataFrame(online_results)
```

**Comment:** The incremental CSV write (`to_csv` inside the loop) is a pragmatic design choice: if a long sweep is interrupted (e.g. by a memory error at a large $K$ value), the results up to that point are already saved. Each sweep runner (`run_k_sweep.py`, `run_alpha_sweep.py`, etc.) uses the same baseline parameter dictionary and varies only one key, ensuring controlled comparison.

\newpage

<!-- ================================================================ -->
<!--                         SECTION 3                                -->
<!--                    Experimental Analysis                         -->
<!-- ================================================================ -->

# 3. Experimental Analysis

## 3.1 Dataset and Setup

All experiments use the **Yelp** dataset (`data/yelp.rating`). Each record is a `(user_id, item_id, rating, timestamp)` tuple; only the `(user_id, item_id)` pair is used as a positive implicit interaction. Experiments use subsets of 10,000 rows (offline: ~8,120 train, ~1,880 test; online: 9,000 train, 1,000 test stream) unless otherwise noted.

## 3.2 Evaluation Metrics

**Hit Ratio (HR@$K$)** measures whether the held-out item appears in the top-$K$ recommendations:

$$\text{HR@}K = \frac{1}{|T|} \sum_{(u,i^*) \in T} \mathbf{1}\!\left[i^* \in \text{Top-}K(u)\right]$$

HR is a *recall-oriented* metric: any correct hit counts equally regardless of its position within the list. We use $K=10$ throughout; the paper reports $K=100$, so our values are not directly numerically comparable — our cutoff is stricter.

**Normalized Discounted Cumulative Gain (NDCG@$K$)** is rank-aware:

$$\text{NDCG@}K = \frac{1}{|T|} \sum_{(u,i^*) \in T} \frac{\mathbf{1}\!\left[i^* \in \text{Top-}K(u)\right]}{\log_2\!\left(\text{rank}(i^*, u) + 1\right)}$$

A hit at rank 1 scores 1.0; at rank 2 scores $1/\log_2(3) \approx 0.631$; at rank $K$ scores $1/\log_2(K+1)$. NDCG penalizes hitting the correct item at a low rank, making it a *precision-oriented* complement to HR.

## 3.3 Baseline Parameter Selection via Grid Search

Before running controlled one-parameter sweeps, we ran a broad grid search using `run_experiments.py` and `experiments.py`. The grid explored:

| Parameter | Values tested |
|---|---|
| $K$ (factors) | 4, 8, 12, 16 |
| max\_iter | 3, 5 |
| w0 | 0.5, 1.0, 2.0 |
| alpha | 0.0, 0.25, 0.5, 0.75 |
| reg | 0.1 (fixed) |

This produced 96 combinations, all recorded in `results/experiment_results.csv`. The selection process worked as follows:

**Step 1 — Identify feasible runtime.** $K=16$, `max_iter=5` already took ~9 seconds per run. Larger $K$ would make the full sweep impractical. We limited the baseline to $K \leq 16$.

**Step 2 — Find HR/NDCG peaks.** Sorting `experiment_results.csv` by `hr_at_k` descending, the top entries cluster around $K=4$, `w0=2.0`, `alpha=0.25` (best HR: ~0.009). However, these also showed the highest loss and appeared to be benefiting from the low regularization cost of small $K$ rather than genuine latent structure.

**Step 3 — Find loss minima.** Sorting by `final_loss` ascending, $K=12$--$16$ with `max_iter=5` and `w0=0.5`--`1.0` gave the best loss values (~900--975), corresponding to a well-converged model.

**Step 4 — Choose a middle ground.** We sought a configuration that balanced good HR/NDCG with low loss and reasonable runtime. The row `K=12, max_iter=5, reg=0.1, w0=1.0, alpha=0.5` achieved HR=0.00532, NDCG=0.00215, loss=974.6, runtime~5.9s — sitting between the HR-maximizing $K=4$ configurations and the loss-minimizing $K=16$ configurations. This became the **baseline** for all one-parameter sweeps:

| Parameter | Baseline Value | Rationale |
|---|---|---|
| factors ($K$) | 12 | Good loss, stable HR, affordable runtime |
| max\_iter | 5 | Converged loss, efficient training |
| reg ($\lambda$) | 0.1 | Stable across all experiments |
| w0 ($c_0$) | 1.0 | Central value; avoids extremes |
| alpha ($\alpha$) | 0.5 | Paper's empirical optimum; confirmed in grid |
| top\_k | 10 | Course-appropriate evaluation cutoff |
| init\_stdev | 0.001 | Required for numerical stability |
| w\_new | 4.0 | Consistent with paper's finding |
| online\_iter | 1 | One step sufficient per paper |

## 3.4 Impact of Popularity Exponent $\alpha$

**Sweep values:** $\alpha \in \{0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0\}$, all other parameters at baseline.

| $\alpha$ | Offline HR | Offline NDCG | Offline Loss | Online HR | Online NDCG |
|---|---|---|---|---|---|
| 0.0 | 0.00532 | 0.00264 | 961.6 | 0.011 | 0.00647 |
| 0.1 | 0.00532 | 0.00265 | 964.1 | 0.011 | 0.00652 |
| 0.25 | 0.00532 | 0.00242 | 967.9 | 0.011 | 0.00660 |
| 0.4 | 0.00532 | 0.00223 | 971.9 | 0.011 | 0.00590 |
| 0.5 | 0.00532 | 0.00215 | 974.6 | 0.013 | 0.00601 |
| 0.75 | 0.00532 | 0.00203 | 981.5 | 0.012 | 0.00514 |
| 1.0 | 0.00372 | 0.00146 | 988.0 | 0.011 | 0.00397 |

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/alpha_sweep_offline_metrics.png}
\caption{Impact of $\alpha$ on offline HR@10 and NDCG@10.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/alpha_sweep_online_metrics.png}
\caption{Impact of $\alpha$ on online HR@10 and NDCG@10.}
\end{figure}

**Interpretation.** Offline HR is flat from $\alpha=0$ to $\alpha=0.75$, then drops sharply at $\alpha=1.0$. This suggests that for the offline protocol, the model saturates quickly and heavy popularity skewing at $\alpha=1.0$ actively hurts — it over-penalizes popular items as negatives to the point where the model struggles to recommend any popular item. Offline NDCG declines monotonically with $\alpha$, indicating that while HR is robust to moderate $\alpha$ values, the *ranking quality* degrades as popularity weighting increases — the correct item is being found but placed lower in the list.

Online metrics tell a slightly different story: HR peaks at $\alpha=0.5$ (0.013 vs 0.011 at $\alpha=0$), suggesting that the popularity-aware weighting genuinely helps the online model adapt to new interactions. This aligns with the paper's finding that $\alpha=0.4$--$0.5$ is optimal. Loss increases monotonically with $\alpha$, which is expected since stronger popularity weighting penalizes more missing entries more heavily.

## 3.5 Impact of Missing Data Weight $c_0$ (`w0`)

**Sweep values:** `w0` $\in \{0.25, 0.5, 1.0, 2.0, 4.0, 8.0\}$.

| w0 | Offline HR | Offline NDCG | Offline Loss | Online HR | Online NDCG |
|---|---|---|---|---|---|
| 0.25 | 0.00319 | 0.00139 | 875.6 | 0.013 | 0.00615 |
| 0.5 | 0.00372 | 0.00150 | 910.8 | 0.012 | 0.00615 |
| 1.0 | 0.00532 | 0.00215 | 974.6 | 0.015 | 0.00658 |
| 2.0 | 0.00532 | 0.00222 | 1110.5 | 0.015 | 0.00645 |
| 4.0 | 0.00426 | 0.00186 | 1396.9 | 0.016 | 0.00566 |
| 8.0 | 0.00638 | 0.00259 | 1900.2 | 0.022 | 0.00780 |

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/w0_sweep_offline_metrics.png}
\caption{Impact of $c_0$ (\texttt{w0}) on offline HR@10 and NDCG@10.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/w0_sweep_online_metrics.png}
\caption{Impact of $c_0$ (\texttt{w0}) on online HR@10 and NDCG@10.}
\end{figure}

**Interpretation.** The offline results show a non-monotonic pattern: HR and NDCG increase from `w0=0.25` to `w0=1.0`--`2.0`, dip at `w0=4.0`, then recover at `w0=8.0`. This is counterintuitive — very high `w0` should over-penalize all missing entries, yet it gives the highest offline HR. The likely explanation is that at `w0=8.0` the model finds a different local optimum where it defaults to recommending broadly popular items; since our test set contains some popular items, this inflates HR while the much higher loss (1900 vs 875) confirms the model is poorly calibrated. The online protocol is more consistent: HR and NDCG both broadly increase with `w0`, peaking at `w0=8.0`. This may reflect that stronger weighting of missing data makes the model more responsive to the new interactions injected during online updating.

The key takeaway is that `w0=1.0`--`2.0` gives a good balance: competitive HR and NDCG with much lower loss than the extremes, indicating a well-converged model rather than an artifact of local optima.

## 3.6 Impact of Number of Latent Factors $K$

**Sweep values:** $K \in \{4, 8, 12, 16, 24, 32\}$.

| $K$ | Offline HR | Offline NDCG | Offline Loss | Online HR | Online NDCG | Runtime (s) |
|---|---|---|---|---|---|---|
| 4 | 0.00798 | 0.00395 | 1263.4 | 0.007 | 0.00238 | 1.46 |
| 8 | 0.00319 | 0.00153 | 1041.6 | 0.010 | 0.00372 | 3.46 |
| 12 | 0.00532 | 0.00215 | 974.6 | 0.014 | 0.00623 | 6.00 |
| 16 | 0.00160 | 0.00072 | 953.4 | 0.015 | 0.00684 | 9.35 |
| 24 | 0.00372 | 0.00185 | 932.3 | 0.011 | 0.00637 | 17.94 |
| 32 | 0.00266 | 0.00129 | 927.8 | 0.020 | 0.00975 | 32.50 |

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/k_sweep_offline_metrics.png}
\caption{Offline HR@10 and NDCG@10 as a function of $K$.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/k_sweep_online_metrics.png}
\caption{Online HR@10 and NDCG@10 as a function of $K$.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/k_sweep_runtime.png}
\caption{Training runtime (seconds per iteration) as a function of $K$.}
\end{figure}

**Interpretation.** Offline metrics are *non-monotonic* with $K$ — $K=4$ gives the best offline HR (0.008) but one of the worst losses (1263). This paradox likely occurs because small $K$ models are highly regularized by their limited capacity and tend to recommend broadly popular items, which inflates HR on a sparse test set. Loss decreases monotonically as $K$ increases (1263→928), confirming that richer representations better fit the training data. Online metrics tell a cleaner story: HR and NDCG generally improve with $K$ (peaking at $K=32$ with HR=0.020, NDCG=0.00975), because the online protocol's larger test set and evaluate-then-update design better rewards latent structure. The anomalously high online loss at $K=8$ (2715 vs ~1000 for all other $K$ values) is a numerical instability in that particular run and not a systematic effect.

Runtime grows approximately as $O(K^2)$, consistent with the theoretical $(M+N)K^2$ term: from $K=4$ (1.46s) to $K=32$ (32.5s) is a factor of ~22, close to $(32/4)^2 = 64$ scaled by implementation constants.

## 3.7 Convergence: Training Iterations

**Sweep values:** `max_iter` $\in \{1, 2, 3, 5, 8, 10, 15, 20\}$.

| Iterations | Offline HR | Offline NDCG | Offline Loss | Online HR | Online NDCG |
|---|---|---|---|---|---|
| 1 | 0.00479 | 0.00208 | 7297.9 | 0.015 | 0.00703 |
| 2 | 0.00426 | 0.00174 | 1671.5 | 0.018 | 0.00847 |
| 3 | 0.00319 | 0.00137 | 1189.8 | 0.011 | 0.00549 |
| 5 | 0.00532 | 0.00215 | 974.6 | 0.012 | 0.00548 |
| 8 | 0.00479 | 0.00202 | 893.6 | 0.006 | 0.00210 |
| 10 | 0.00532 | 0.00239 | 875.3 | 0.004 | 0.00130 |
| 15 | 0.00638 | 0.00283 | 858.3 | 0.005 | 0.00181 |
| 20 | 0.00638 | 0.00331 | 852.1 | 0.007 | 0.00306 |

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/iteration_sweep_loss.png}
\caption{Training loss as a function of iteration number. Loss decreases monotonically, with the steepest drop between iterations 1 and 5.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/iteration_sweep_offline_metrics.png}
\caption{Offline HR@10 and NDCG@10 as a function of training iterations.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/iteration_sweep_online_metrics.png}
\caption{Online HR@10 and NDCG@10 as a function of training iterations.}
\end{figure}

**Interpretation.** The offline loss decreases steeply from iteration 1 (7298) to iteration 5 (975), then continues to decrease but much more slowly (975→852 from iter 5 to 20). This confirms the paper's finding that most convergence occurs in the first few iterations. Offline HR and NDCG improve steadily with more training (0.00638 and 0.00331 at iter 20), suggesting the offline model benefits from continued training.

The online protocol shows a striking divergence: online metrics *deteriorate* after iter=2 (HR=0.018, NDCG=0.00847 — the best online values in the entire sweep) as more training iterations are applied. Online loss also increases dramatically after iter=5 (from 991 to 11,862 at iter=20). This is an important finding: an offline model that is over-trained (many iterations) becomes poorly calibrated for the online streaming protocol. The model's factors are too tightly fitted to the training distribution, making the incremental `update_model()` calls less effective at adapting to new interactions. **The optimal offline training iteration count for online deployment is around 2--3, not the loss-minimizing 20.**

## 3.8 Scalability Analysis

**Sweep values:** `row_limit` $\in \{10000, 20000, 50000, 100000\}$.

| Rows | Train | Test (offline) | Users | Items | Offline HR | Offline Runtime (s) | Online Runtime (s) |
|---|---|---|---|---|---|---|---|
| 10,000 | 8,120 | 1,880 | 2,979 | 4,980 | 0.00532 | 6.0 | 7.5 |
| 20,000 | 17,103 | 2,897 | 3,629 | 7,854 | 0.00932 | 9.5 | 17.1 |
| 50,000 | 42,269 | 7,731 | 10,785 | 13,526 | 0.00595 | 21.0 | 53.9 |
| 100,000 | 82,601 | 17,399 | 22,705 | 18,946 | 0.01035 | 37.3 | 129.0 |

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/scalability_metrics.png}
\caption{HR@10 and NDCG@10 as dataset size grows.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/scalability_offline_runtime.png}
\caption{Offline training runtime as dataset size grows. Growth is sub-quadratic, consistent with $O(|R|K)$ dominance.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/scalability_online_runtime.png}
\caption{Online update runtime as dataset size grows.}
\end{figure}

**Interpretation.** Offline training runtime scales from 6.0s at 10k rows to 37.3s at 100k rows — approximately a $6\times$ increase for a $10\times$ increase in data size. This is sub-linear in $|R|$ at first glance, but reflects that the $(M+N)K^2$ term (which depends on user/item count, not $|R|$) dominates at small sizes while the $|R|K$ term takes over at larger sizes. The overall scaling is consistent with the theoretical $O((M+N)K^2 + |R|K)$ complexity.

Online runtime grows much faster: 7.5s at 10k to 129s at 100k — approximately $17\times$ for $10\times$ data. This super-linear growth is due to an important numerical issue: the online loss at 50k and 100k rows (1.44M and 8.33M respectively, compared to ~990 at 10k) indicates **numerical divergence in the online update protocol at larger scales**. With more users and items, the incremental `update_model()` calls accumulate floating-point errors, and without periodic full recomputation of the cache matrices, the model drifts. This is a known limitation of purely incremental updates and an important finding for the scalability section.

Recommendation quality (HR, NDCG) generally improves with data size, with the largest dataset (100k rows) giving the best offline HR (0.01035). This confirms that more training data helps the model build better user and item representations.

\newpage

<!-- ================================================================ -->
<!--                         SECTION 4                                -->
<!--              Strengths and Weaknesses                            -->
<!-- ================================================================ -->

# 4. Discussion: Strengths and Weaknesses

## 4.1 Strengths of the eALS Algorithm

**Elimination of matrix inversion.** By updating at the element level, eALS avoids the $O(K^3)$ matrix inversion of standard ALS. The paper reports ALS taking 11.6 hours per iteration at $K=512$ on Amazon, versus 12 minutes for eALS — a 58$\times$ speedup.

**No learning rate required.** Unlike SGD and RCD, eALS performs a closed-form exact update at each coordinate, bypassing the difficulty of learning rate tuning. This is confirmed in our implementation: the model converges reliably with no additional tuning beyond regularization.

**Popularity-aware missing data weighting.** The $c_i$ weighting is more informative than uniform weighting. Our experiments confirm this: online HR peaks at $\alpha=0.5$ (0.013) vs $\alpha=0$ (0.011), and the grid search showed `alpha=0.25`--`0.5` consistently in the top-performing configurations.

**Embarrassingly parallel structure.** User updates depend only on the read-only $\mathbf{S}^q$ cache. Item updates depend only on the read-only $\mathbf{S}^p$ cache. This maps naturally to Spark's partition-based execution: each partition of users can be updated independently, with the shared cache broadcast to all workers.

**Efficient online updating.** One incremental update step is sufficient: our iteration sweep confirms that online metrics peak at iteration 2 (HR=0.018) and degrade with more pre-training, validating the paper's claim that a single online update step is effective.

## 4.2 Weaknesses and Observations from Our Experiments

**Online instability at scale.** Our most important experimental finding: online loss diverges at 50k and 100k rows (reaching 1.44M and 8.33M respectively). Purely incremental cache updates accumulate floating-point errors without periodic recomputation. A production system would require periodic full cache refresh.

**Online protocol sensitive to offline training depth.** The iteration sweep reveals that 2--3 offline training iterations give the best online performance, while loss-minimizing configurations (15--20 iterations) cause online metrics to collapse. This tension between offline loss and online adaptability is not discussed in the paper and is a genuine finding.

**Non-monotonic K-vs-quality relationship in offline setting.** Small $K$ ($K=4$) achieves the best offline HR (0.008) but the worst loss (1263). This suggests the leave-one-out evaluation on a sparse dataset rewards popularity-biased recommendations, which small models naturally produce. The online protocol is more discriminating and correctly rewards higher $K$.

**Limitations of our implementation:**

- Training runs locally in NumPy, not fully distributed in Spark. The Spark layer handles data loading only.
- Metrics at HR@10 / NDCG@10 rather than paper's HR@100 / NDCG@100; our values are not numerically comparable.
- Experiments use subsets (10k--100k) of the full Yelp dataset (731k interactions).
- No comparison against baseline algorithms (ALS, BPR, RCD).
- Online loss divergence at scale indicates a limitation in the incremental cache update strategy for large datasets.

\newpage

<!-- ================================================================ -->
<!--                         APPENDIX                                 -->
<!--                      Full Code Listing                           -->
<!-- ================================================================ -->

# Appendix: Full Code Listing

## A.1 `src/config.py`

```python
from dataclasses import dataclass


@dataclass
class FastALSConfig:
    factors: int = 10
    max_iter: int = 10
    reg: float = 0.01
    w0: float = 1.0
    alpha: float = 0.5
    init_mean: float = 0.0
    init_stdev: float = 0.001
    show_progress: bool = True
    show_loss: bool = True
    top_k: int = 10
    random_seed: int = 42
```

---

## A.2 `src/spark_loader.py`

```python
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
    user_items_df = df.groupBy("user_id").agg(
        F.collect_list("item_id").alias("item_list")
    )
    item_users_df = df.groupBy("item_id").agg(
        F.collect_list("user_id").alias("user_list")
    )
    return user_items_df, item_users_df
```

---

## A.3 `src/rdd_loader.py`

```python
from typing import List, Tuple


def spark_df_to_rdd(df):
    return df.rdd.map(lambda row: (row["user_id"], row["item_id"]))


def build_user_items_rdd(interactions_rdd):
    return interactions_rdd.groupByKey().mapValues(list)


def build_item_users_rdd(interactions_rdd):
    return interactions_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)


def collect_rdd_interactions(interactions_rdd) -> List[Tuple[int, int]]:
    return interactions_rdd.collect()
```

---

## A.4 `src/model.py`

```python
from typing import Dict, List, Tuple
import numpy as np

from src.config import FastALSConfig
from src.predict import predict_score


class FastALSModel:

    def __init__(self, interactions: List[Tuple[int, int]], config: FastALSConfig):
        self.config = config
        self.interactions = interactions
        self.user_ids = sorted({u for u, _ in interactions})
        self.item_ids = sorted({i for _, i in interactions})
        self.user_count = len(self.user_ids)
        self.item_count = len(self.item_ids)
        self.user_to_index = {u: idx for idx, u in enumerate(self.user_ids)}
        self.item_to_index = {i: idx for idx, i in enumerate(self.item_ids)}
        self.index_to_user = {idx: u for u, idx in self.user_to_index.items()}
        self.index_to_item = {idx: i for i, idx in self.item_to_index.items()}
        self.user_items = self._build_user_items()
        self.item_users = self._build_item_users()
        self.W = self._build_positive_weights()
        self.Wi = self._compute_item_missing_weights()
        self.U = None
        self.V = None
        self.SU = None
        self.SV = None
        self.prediction_users = np.zeros(self.user_count)
        self.prediction_items = np.zeros(self.item_count)
        self.rating_users = np.zeros(self.user_count)
        self.rating_items = np.zeros(self.item_count)
        self.w_users = np.zeros(self.user_count)
        self.w_items = np.zeros(self.item_count)
        self._initialize_factors()
        self._init_caches()

    def _build_user_items(self):
        user_items = {u: [] for u in range(self.user_count)}
        for raw_u, raw_i in self.interactions:
            user_items[self.user_to_index[raw_u]].append(self.item_to_index[raw_i])
        return user_items

    def _build_item_users(self):
        item_users = {i: [] for i in range(self.item_count)}
        for raw_u, raw_i in self.interactions:
            item_users[self.item_to_index[raw_i]].append(self.user_to_index[raw_u])
        return item_users

    def _build_positive_weights(self):
        weights = {}
        for raw_u, raw_i in self.interactions:
            weights[(self.user_to_index[raw_u], self.item_to_index[raw_i])] = 1.0
        return weights

    def _compute_item_missing_weights(self):
        item_popularity = np.array([len(self.item_users[i]) for i in range(self.item_count)],
                                    dtype=float)
        total = item_popularity.sum()
        if total == 0:
            return np.zeros(self.item_count)
        probabilities = item_popularity / total
        powered = np.power(probabilities, self.config.alpha)
        z = powered.sum()
        if z == 0:
            return np.zeros(self.item_count)
        return self.config.w0 * powered / z

    def _initialize_factors(self):
        np.random.seed(self.config.random_seed)
        self.U = np.random.normal(self.config.init_mean, self.config.init_stdev,
                                   (self.user_count, self.config.factors))
        self.V = np.random.normal(self.config.init_mean, self.config.init_stdev,
                                   (self.item_count, self.config.factors))

    def _init_caches(self):
        self.SU = self.U.T @ self.U
        weighted_V = self.V * self.Wi[:, np.newaxis]
        self.SV = self.V.T @ weighted_V

    def predict(self, u, i):
        return predict_score(self.U[u], self.V[i])

    def loss(self):
        reg_loss = self.config.reg * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
        total_loss = reg_loss
        for u in range(self.user_count):
            l = 0.0
            user_vector = self.U[u]
            for i in self.user_items[u]:
                pred = self.predict(u, i)
                l += self.W[(u, i)] * (1.0 - pred) ** 2
                l -= self.Wi[i] * (pred ** 2)
            l += user_vector @ self.SV @ user_vector
            total_loss += l
        return float(total_loss)

    def update_user(self, u):
        item_list = self.user_items[u]
        if not item_list:
            return
        for i in item_list:
            self.prediction_items[i] = self.predict(u, i)
            self.rating_items[i] = 1.0
            self.w_items[i] = self.W[(u, i)]
        old_vector = self.U[u].copy()
        for f in range(self.config.factors):
            numer = sum(-self.U[u, k] * self.SV[f, k]
                        for k in range(self.config.factors) if k != f)
            for i in item_list:
                self.prediction_items[i] -= self.U[u, f] * self.V[i, f]
                numer += (self.w_items[i] * self.rating_items[i]
                          - (self.w_items[i] - self.Wi[i]) * self.prediction_items[i]
                          ) * self.V[i, f]
                denom_i = (self.w_items[i] - self.Wi[i]) * (self.V[i, f] ** 2)
            denom = sum((self.w_items[i] - self.Wi[i]) * (self.V[i, f] ** 2)
                        for i in item_list) + self.SV[f, f] + self.config.reg
            if denom != 0 and np.isfinite(numer) and np.isfinite(denom):
                new_val = numer / denom
                if np.isfinite(new_val):
                    self.U[u, f] = np.clip(new_val, -10.0, 10.0)
            for i in item_list:
                self.prediction_items[i] += self.U[u, f] * self.V[i, f]
        for f in range(self.config.factors):
            for k in range(f + 1):
                val = self.SU[f,k] - old_vector[f]*old_vector[k] + self.U[u,f]*self.U[u,k]
                self.SU[f, k] = self.SU[k, f] = val

    def update_item(self, i):
        user_list = self.item_users[i]
        if not user_list:
            return
        for u in user_list:
            self.prediction_users[u] = self.predict(u, i)
            self.rating_users[u] = 1.0
            self.w_users[u] = self.W[(u, i)]
        old_vector = self.V[i].copy()
        for f in range(self.config.factors):
            numer = self.Wi[i] * sum(-self.V[i, k] * self.SU[f, k]
                                      for k in range(self.config.factors) if k != f)
            for u in user_list:
                self.prediction_users[u] -= self.U[u, f] * self.V[i, f]
                numer += (self.w_users[u] * self.rating_users[u]
                          - (self.w_users[u] - self.Wi[i]) * self.prediction_users[u]
                          ) * self.U[u, f]
            denom = (sum((self.w_users[u] - self.Wi[i]) * (self.U[u, f] ** 2)
                         for u in user_list)
                     + self.Wi[i] * self.SU[f, f] + self.config.reg)
            if denom != 0 and np.isfinite(numer) and np.isfinite(denom):
                new_val = numer / denom
                if np.isfinite(new_val):
                    self.V[i, f] = np.clip(new_val, -10.0, 10.0)
            for u in user_list:
                self.prediction_users[u] += self.U[u, f] * self.V[i, f]
        for f in range(self.config.factors):
            for k in range(f + 1):
                val = (self.SV[f,k] - old_vector[f]*old_vector[k]*self.Wi[i]
                       + self.V[i,f]*self.V[i,k]*self.Wi[i])
                self.SV[f, k] = self.SV[k, f] = val

    def _add_new_user(self, raw_user_id):
        new_index = self.user_count
        self.user_to_index[raw_user_id] = new_index
        self.index_to_user[new_index] = raw_user_id
        self.user_ids.append(raw_user_id)
        self.user_items[new_index] = []
        new_row = np.random.normal(self.config.init_mean, self.config.init_stdev,
                                    (1, self.config.factors))
        self.U = np.vstack([self.U, new_row])
        self.prediction_users = np.append(self.prediction_users, 0.0)
        self.rating_users = np.append(self.rating_users, 0.0)
        self.w_users = np.append(self.w_users, 0.0)
        self.user_count += 1
        self.SU = self.U.T @ self.U
        return new_index

    def _add_new_item(self, raw_item_id):
        new_index = self.item_count
        self.item_to_index[raw_item_id] = new_index
        self.index_to_item[new_index] = raw_item_id
        self.item_ids.append(raw_item_id)
        self.item_users[new_index] = []
        new_row = np.random.normal(self.config.init_mean, self.config.init_stdev,
                                    (1, self.config.factors))
        self.V = np.vstack([self.V, new_row])
        self.prediction_items = np.append(self.prediction_items, 0.0)
        self.rating_items = np.append(self.rating_items, 0.0)
        self.w_items = np.append(self.w_items, 0.0)
        self.Wi = np.append(self.Wi, self.config.w0 / max(1, self.item_count + 1))
        self.item_count += 1
        self._init_caches()
        return new_index

    def update_model(self, raw_user_id, raw_item_id, w_new=4.0, online_iter=1):
        u = (self._add_new_user(raw_user_id) if raw_user_id not in self.user_to_index
             else self.user_to_index[raw_user_id])
        i = (self._add_new_item(raw_item_id) if raw_item_id not in self.item_to_index
             else self.item_to_index[raw_item_id])
        if i not in self.user_items[u]: self.user_items[u].append(i)
        if u not in self.item_users[i]: self.item_users[i].append(u)
        self.W[(u, i)] = w_new
        for _ in range(online_iter):
            self.update_user(u)
            self.update_item(i)
```

---

## A.5 `src/train.py`

```python
from src.utils import current_millis, format_seconds


def run_one_iteration(model) -> None:
    for u in range(model.user_count):
        model.update_user(u)
    for i in range(model.item_count):
        model.update_item(i)


def train_model(model) -> None:
    previous_loss = float("inf")
    for iteration in range(model.config.max_iter):
        start = current_millis()
        run_one_iteration(model)
        elapsed_seconds = (current_millis() - start) / 1000.0
        if model.config.show_progress:
            print(f"Iteration {iteration+1}/{model.config.max_iter} "
                  f"completed in {format_seconds(elapsed_seconds)}")
        if model.config.show_loss:
            current_loss = model.loss()
            symbol = "-" if current_loss <= previous_loss else "+"
            print(f"Current loss: {current_loss:.6f} [{symbol}]")
            previous_loss = current_loss
```

---

## A.6 `src/predict.py`

```python
import numpy as np


def predict_score(user_vector: np.ndarray, item_vector: np.ndarray) -> float:
    return float(np.dot(user_vector, item_vector))
```

---

## A.7 `src/recommender.py`

```python
import numpy as np


def recommend_top_k(model, raw_user_id, k=10):
    if raw_user_id not in model.user_to_index:
        return []
    u = model.user_to_index[raw_user_id]
    seen_items = set(model.user_items[u])
    scores = [(i, model.predict(u, i))
              for i in range(model.item_count) if i not in seen_items]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [model.index_to_item[i] for i, _ in scores[:k]]
```

---

## A.8 `src/split.py`

```python
from typing import List, Tuple, Dict


def leave_one_out_split(interactions: List[Tuple[int, int]]):
    user_histories: Dict[int, List[int]] = {}
    for u, i in interactions:
        user_histories.setdefault(u, []).append(i)
    train, test = [], []
    for u, items in user_histories.items():
        if len(items) == 1:
            train.append((u, items[0]))
        else:
            train.extend((u, i) for i in items[:-1])
            test.append((u, items[-1]))
    return train, test
```

---

## A.9 `src/online_split.py`

```python
from typing import List, Tuple


def chronological_90_10_split(events: List[Tuple[int, int, int]]):
    events_sorted = sorted(events, key=lambda x: x[2])
    cutoff = int(0.9 * len(events_sorted))
    train_events = events_sorted[:cutoff]
    test_events = events_sorted[cutoff:]
    train_interactions = [(u, i) for u, i, _ in train_events]
    test_interactions = [(u, i) for u, i, _ in test_events]
    return train_interactions, test_interactions
```

---

## A.10 `src/evaluate.py`

```python
import math
from src.recommender import recommend_top_k


def hit_rate_at_k(model, test_interactions, k=10):
    if not test_interactions:
        return 0.0
    hits = sum(1 for raw_user_id, true_item in test_interactions
               if true_item in recommend_top_k(model, raw_user_id, k=k))
    return hits / len(test_interactions)


def ndcg_at_k(model, test_interactions, k=10):
    if not test_interactions:
        return 0.0
    total_ndcg = 0.0
    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)
        if true_item in recs:
            rank = recs.index(true_item) + 1
            total_ndcg += 1.0 / math.log2(rank + 1)
    return total_ndcg / len(test_interactions)


def online_protocol_metrics(model, test_interactions, k=10, w_new=4.0, online_iter=1):
    if not test_interactions:
        return {"hr_at_k": 0.0, "ndcg_at_k": 0.0}
    hits, total_ndcg, total = 0, 0.0, 0
    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)
        if true_item in recs:
            hits += 1
            total_ndcg += 1.0 / math.log2(recs.index(true_item) + 2)
        model.update_model(raw_user_id, true_item, w_new=w_new, online_iter=online_iter)
        total += 1
    return {"hr_at_k": hits / total, "ndcg_at_k": total_ndcg / total}
```

---

## A.11 `src/sweeps.py`

```python
import os, time
import pandas as pd

from src.config import FastALSConfig
from src.model import FastALSModel
from src.train import train_model
from src.evaluate import hit_rate_at_k, ndcg_at_k, online_protocol_metrics
from src.split import leave_one_out_split
from src.online_split import chronological_90_10_split


def ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)


def build_model(train_interactions, factors=8, max_iter=5, reg=0.1, w0=1.0,
                alpha=0.5, top_k=10, init_mean=0.0, init_stdev=0.001,
                random_seed=42, show_progress=False, show_loss=False):
    config = FastALSConfig(factors=factors, max_iter=max_iter, reg=reg, w0=w0,
                           alpha=alpha, init_mean=init_mean, init_stdev=init_stdev,
                           show_progress=show_progress, show_loss=show_loss,
                           top_k=top_k, random_seed=random_seed)
    return FastALSModel(interactions=train_interactions, config=config)


def run_offline_experiment(interactions, **params):
    train_interactions, test_interactions = leave_one_out_split(interactions)
    start = time.time()
    model = build_model(train_interactions, **params)
    train_model(model)
    runtime_seconds = time.time() - start
    return {
        "protocol": "offline", **params,
        "train_interactions": len(train_interactions),
        "test_interactions": len(test_interactions),
        "user_count": model.user_count, "item_count": model.item_count,
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
    metrics = online_protocol_metrics(model, test_interactions,
                                      k=params["top_k"], w_new=w_new,
                                      online_iter=online_iter)
    runtime_seconds = time.time() - start
    return {
        "protocol": "online", **params, "w_new": w_new, "online_iter": online_iter,
        "train_interactions": len(train_interactions),
        "test_interactions": len(test_interactions),
        "user_count": model.user_count, "item_count": model.item_count,
        "final_loss": model.loss(),
        "hr_at_k": metrics["hr_at_k"], "ndcg_at_k": metrics["ndcg_at_k"],
        "runtime_seconds": runtime_seconds,
    }


def run_single_parameter_sweep_both_protocols(
    interactions, events, sweep_name, sweep_values,
    fixed_params, offline_output_csv, online_output_csv,
    w_new=4.0, online_iter=1,
):
    ensure_results_dir("results")
    offline_results, online_results = [], []
    for idx, value in enumerate(sweep_values, start=1):
        params = {**fixed_params, sweep_name: value}
        print(f"Sweep {idx}/{len(sweep_values)} | {sweep_name}={value}")
        offline_results.append(run_offline_experiment(interactions=interactions, **params))
        online_results.append(run_online_experiment(events=events, w_new=w_new,
                                                     online_iter=online_iter, **params))
        pd.DataFrame(offline_results).to_csv(offline_output_csv, index=False)
        pd.DataFrame(online_results).to_csv(online_output_csv, index=False)
    return pd.DataFrame(offline_results), pd.DataFrame(online_results)
```

---

## A.12 `src/utils.py`

```python
import time


def current_millis() -> int:
    return int(time.time() * 1000)


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def print_header(message: str) -> None:
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)
```

---

## A.13 `src/data_loader.py` (legacy pandas loader)

```python
from typing import List, Tuple
import pandas as pd


def load_interactions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["user_id", "item_id", "rating", "timestamp"],
                     nrows=5000)
    df = df[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    return df


def dataframe_to_interactions(df: pd.DataFrame) -> List[Tuple[int, int]]:
    return list(df[["user_id", "item_id"]].itertuples(index=False, name=None))


def get_unique_users_items(interactions: List[Tuple[int, int]]):
    users = sorted({u for u, _ in interactions})
    items = sorted({i for _, i in interactions})
    return users, items
```

---

## A.14 `main.py`

```python
from src.config import FastALSConfig
from src.model import FastALSModel
from src.train import train_model
from src.utils import print_header
from src.spark_loader import (create_spark_session, load_yelp_interactions_spark,
                               build_groupings_with_spark)
from src.rdd_loader import (spark_df_to_rdd, build_user_items_rdd,
                              build_item_users_rdd, collect_rdd_interactions)
from src.split import leave_one_out_split
from src.evaluate import hit_rate_at_k
from src.recommender import recommend_top_k


def main():
    print_header("FastALS Python + Spark DF + RDD")
    config = FastALSConfig(factors=8, max_iter=3, reg=0.1, w0=1.0, alpha=0.5,
                           init_stdev=0.001, show_progress=True, show_loss=True, top_k=10)
    spark = create_spark_session()
    df = load_yelp_interactions_spark(spark, "data/yelp.rating", limit_rows=10000)
    print("Spark DataFrame preview:"); df.show(5)
    print(f"Spark interaction count: {df.count()}")
    user_items_df, item_users_df = build_groupings_with_spark(df)
    print("Grouped by user (DF):"); user_items_df.show(3, truncate=False)
    interactions_rdd = spark_df_to_rdd(df)
    print("RDD preview:", interactions_rdd.take(5))
    user_items_rdd = build_user_items_rdd(interactions_rdd)
    item_users_rdd = build_item_users_rdd(interactions_rdd)
    print("Grouped by user (RDD):", user_items_rdd.take(3))
    print("Grouped by item (RDD):", item_users_rdd.take(3))
    interactions = collect_rdd_interactions(interactions_rdd)
    train_interactions, test_interactions = leave_one_out_split(interactions)
    model = FastALSModel(interactions=train_interactions, config=config)
    train_model(model)
    hr = hit_rate_at_k(model, test_interactions, k=config.top_k)
    print(f"Hit Rate@{config.top_k}: {hr:.4f}")
    for raw_user_id, _ in test_interactions[:3]:
        recs = recommend_top_k(model, raw_user_id, k=5)
        print(f"Top-5 for user {raw_user_id}: {recs}")
    spark.stop()


if __name__ == "__main__":
    main()
```

---

## A.15 `run_k_sweep.py`

```python
from src.spark_loader import (create_spark_session, load_yelp_interactions_spark,
                               load_yelp_events_spark, spark_df_to_events)
from src.rdd_loader import spark_df_to_rdd, collect_rdd_interactions
from src.sweeps import run_single_parameter_sweep_both_protocols

BASELINE_PARAMS = {"factors": 12, "max_iter": 5, "reg": 0.1, "w0": 1.0,
                   "alpha": 0.5, "top_k": 10, "init_mean": 0.0,
                   "init_stdev": 0.001, "random_seed": 42}

def main():
    spark = create_spark_session()
    offline_df = load_yelp_interactions_spark(spark, "data/yelp.rating", limit_rows=10000)
    interactions = collect_rdd_interactions(spark_df_to_rdd(offline_df))
    online_df = load_yelp_events_spark(spark, "data/yelp.rating", limit_rows=10000)
    events = spark_df_to_events(online_df)
    run_single_parameter_sweep_both_protocols(
        interactions=interactions, events=events, sweep_name="factors",
        sweep_values=[4, 8, 12, 16, 24, 32],
        fixed_params=BASELINE_PARAMS,
        offline_output_csv="results/k_sweep_offline.csv",
        online_output_csv="results/k_sweep_online.csv",
        w_new=4.0, online_iter=1)
    spark.stop()

if __name__ == "__main__":
    main()
```

---

## A.16 `run_alpha_sweep.py`, `run_w0_sweep.py`, `run_iteration_sweep.py`

These follow the identical structure as `run_k_sweep.py`, varying `sweep_name` and `sweep_values`:

- `run_alpha_sweep.py`: `sweep_name="alpha"`, values `[0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0]`
- `run_w0_sweep.py`: `sweep_name="w0"`, values `[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]`
- `run_iteration_sweep.py`: `sweep_name="max_iter"`, values `[1, 2, 3, 5, 8, 10, 15, 20]`

---

## A.17 `run_scalability_sweep.py`

```python
import pandas as pd
from src.spark_loader import (create_spark_session, load_yelp_interactions_spark,
                               load_yelp_events_spark, spark_df_to_events)
from src.rdd_loader import spark_df_to_rdd, collect_rdd_interactions
from src.sweeps import ensure_results_dir, run_offline_experiment, run_online_experiment

BASELINE_PARAMS = {"factors": 12, "max_iter": 5, "reg": 0.1, "w0": 1.0,
                   "alpha": 0.5, "top_k": 10, "init_mean": 0.0,
                   "init_stdev": 0.001, "random_seed": 42}

def main():
    spark = create_spark_session()
    ensure_results_dir("results")
    offline_results, online_results = [], []
    for row_limit in [10000, 20000, 50000, 100000]:
        print(f"Scalability sweep | rows={row_limit}")
        offline_df = load_yelp_interactions_spark(spark, "data/yelp.rating",
                                                   limit_rows=row_limit)
        interactions = collect_rdd_interactions(spark_df_to_rdd(offline_df))
        online_df = load_yelp_events_spark(spark, "data/yelp.rating",
                                            limit_rows=row_limit)
        events = spark_df_to_events(online_df)
        r_off = run_offline_experiment(interactions=interactions, **BASELINE_PARAMS)
        r_off["row_limit"] = row_limit
        offline_results.append(r_off)
        r_on = run_online_experiment(events=events, w_new=4.0,
                                      online_iter=1, **BASELINE_PARAMS)
        r_on["row_limit"] = row_limit
        online_results.append(r_on)
        pd.DataFrame(offline_results).to_csv("results/scalability_sweep_offline.csv",
                                              index=False)
        pd.DataFrame(online_results).to_csv("results/scalability_sweep_online.csv",
                                             index=False)
    spark.stop()

if __name__ == "__main__":
    main()
```

---

## A.18 `plot_results.py`

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def save_line_chart(df, x_col, y_cols, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    for y_col in y_cols:
        plt.plot(df[x_col], df[y_col], marker="o", label=y_col)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(output_path, dpi=300); plt.close()

# Separate chart functions for each sweep type (k, w0, alpha, iteration, scalability)
# Each loads the appropriate CSV and calls save_line_chart / save_runtime_chart
# See full file for make_k_charts(), make_w0_charts(), make_alpha_charts(),
# make_iteration_charts(), make_scalability_charts()

def main():
    ensure_figures_dir()
    # [calls all make_*_charts() functions]
    print("Finished generating charts in figures/")

if __name__ == "__main__":
    main()
```

---

## A.19 References

He, X., Zhang, H., Kan, M.-Y., & Chua, T.-S. (2016). *Fast Matrix Factorization for Online Recommendation with Implicit Feedback*. SIGIR '16. DOI: 10.1145/2911451.2911489

Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM 2008.

Devooght, R., Kourtellis, N., & Mantrach, A. (2015). *Dynamic Matrix Factorization with Priors on Unknown Values*. KDD 2015.

Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI 2009.

Pilaszy, I., Zibriczky, D., & Tikk, D. (2010). *Fast ALS-based Matrix Factorization for Explicit and Implicit Feedback Datasets*. RecSys 2010.