# Fast eALS in PySpark + Python: DBMS Project

## 1. Project overview

This project is a Python / PySpark reimplementation of the algorithm from the paper:

**Fast Matrix Factorization for Online Recommendation with Implicit Feedback**

The original paper presents a fast element-wise ALS approach for matrix factorization on implicit-feedback data, with two major themes:

1. **Non-uniform weighting of missing entries**, based on item popularity
2. **Online incremental updating**, so the model can adapt to new interactions quickly

The original author implementation is in Java.  
Our project translates the main logic into Python, while using **PySpark DataFrames and RDDs** for data loading and preprocessing.

This project is designed for a DBMS experimental analysis assignment, not as a production recommender system.

---

## 2. Main objective of our implementation

The goal of this project is to:

- understand and reimplement the Java FastALS / eALS logic in Python
- integrate Spark into the pipeline for loading and preprocessing data
- run controlled experiments on one of the datasets used in the paper
- evaluate the behavior of the algorithm under different parameters
- compare trends with the paper, not necessarily reproduce exact numbers

---

## 3. Important note on faithfulness to the paper

Our implementation is **structurally faithful** to the paper and the author’s Java code, but not an exact reproduction of the paper’s full experimental environment.

### What is faithful
- latent factor model with user/item matrices `U` and `V`
- popularity-based weighting of missing data via `Wi`
- cache matrices `SU` and `SV`
- element-wise alternating updates for users and items
- incremental online update logic for new interactions
- offline and online experiment protocols

### What is simplified
- model training is still done **locally in NumPy**, not fully distributed across Spark workers
- evaluation uses **HR@10** and **NDCG@10** instead of the paper’s HR/NDCG@100
- datasets are tested on controlled subsets first (5k, 10k, etc.) rather than always the full dataset
- no baseline comparison against other algorithms such as BPR, vanilla ALS, RCD, etc.
- online experiments use a simplified project-level implementation of the online update protocol

This must be stated clearly in the report.

---

## 4. Project development path

The project was developed in stages.

### Stage 1: Java understanding
We began from the author’s Java file `MF_fastALS.java`, which contains the main training logic.  
We also used:
- `main_MF.java`
- `TopKRecommender.java`
- `SparseMatrix.java`
- `SparseVector.java`
- `DenseMatrix.java`
- `DenseVector.java`
- `Rating.java`
- `Printer.java`

From these files, we identified:
- how the model is initialized
- how `Wi` is computed
- how `SU` and `SV` are cached
- how `update_user()` works
- how `update_item()` works
- how the online `updateModel()` works

### Stage 2: Python skeleton
We first created a pure Python project skeleton:
- config file
- model class
- train loop
- prediction
- evaluation placeholders
- data loader

This was done before Spark so that the Java logic could be ported safely.

### Stage 3: Stable local reference implementation
We then implemented:
- `update_user`
- `update_item`
- `loss`

At first, training was numerically unstable.  
We stabilized it by:
- deduplicating `(user, item)` pairs for the offline protocol
- reducing the initialization variance
- adding stronger regularization
- clipping extreme updated values
- validating on smaller subsets first

This gave us a working local reference implementation.

### Stage 4: Spark integration
We added Spark in two layers:
- **DataFrame layer** for file loading, parsing, deduplication, grouping
- **RDD layer** for grouped user/item interaction structures

Spark is therefore used in the preprocessing and data pipeline, while the core matrix updates remain local.

### Stage 5: Evaluation pipeline
We added:
- offline train/test splitting
- top-K recommendation generation
- HR@K
- NDCG@K

### Stage 6: Online protocol
We added:
- chronological event loading with timestamps
- 90/10 chronological split
- evaluate-then-update streaming protocol
- incremental update with `w_new` and `online_iter`

### Stage 7: Experiment automation
We created sweep runners that:
- vary one parameter at a time
- save separate CSVs for offline and online experiments
- support scalability experiments over data size

---

## 5. Project structure and what each file does

### Root-level files

#### `main.py`
Main demo runner for the current end-to-end pipeline.
It:
- loads Yelp data with Spark
- shows DataFrame and RDD previews
- builds a train/test split
- trains the model
- computes HR@10
- prints a few recommendations

#### `run_experiments.py`
Older exploratory grid-search style runner.
Useful for broad parameter exploration but not the main report format.

#### `run_k_sweep.py`
Runs controlled sweeps over the number of latent factors `K`  
Saves:
- `results/k_sweep_offline.csv`
- `results/k_sweep_online.csv`

#### `run_w0_sweep.py`
Runs controlled sweeps over `w0`  
Saves:
- `results/w0_sweep_offline.csv`
- `results/w0_sweep_online.csv`

#### `run_alpha_sweep.py`
Runs controlled sweeps over `alpha`  
Saves:
- `results/alpha_sweep_offline.csv`
- `results/alpha_sweep_online.csv`

#### `run_iteration_sweep.py`
Runs controlled sweeps over `max_iter`  
Saves:
- `results/iteration_sweep_offline.csv`
- `results/iteration_sweep_online.csv`

#### `run_scalability_sweep.py`
Runs scalability experiments over different dataset sizes  
Saves:
- `results/scalability_sweep_offline.csv`
- `results/scalability_sweep_online.csv`

---

### `src/` directory

#### `src/config.py`
Contains the `FastALSConfig` dataclass.  
Holds all main hyperparameters:
- `factors`
- `max_iter`
- `reg`
- `w0`
- `alpha`
- `top_k`
- initialization settings

#### `src/utils.py`
Small utility functions for printing/timing.

#### `src/data_loader.py`
Initial pandas-based loader used during the earlier pure Python stage.  
Kept for reference.

#### `src/spark_loader.py`
Main Spark loader file.

Contains:
- `create_spark_session()`
- `load_yelp_interactions_spark()`
- `load_yelp_events_spark()`
- `spark_df_to_interactions()`
- `spark_df_to_events()`
- `build_groupings_with_spark()`

This file is responsible for:
- reading raw text lines from `yelp.rating`
- splitting lines into user/item/rating/timestamp
- preparing offline interactions
- preparing timestamped events for online experiments

#### `src/rdd_loader.py`
Contains the RDD conversion layer.

Functions include:
- converting DataFrames to RDDs
- grouping user → item list
- grouping item → user list
- collecting interactions back into local Python

This file shows the explicit use of **RDDs** in the project.

#### `src/model.py`
The core model implementation.

Contains:
- model initialization
- user/item ID mapping
- building `user_items` and `item_users`
- positive weights `W`
- negative item weights `Wi`
- initialization of `U` and `V`
- cache initialization `SU`, `SV`
- `predict()`
- `loss()`
- `update_user()`
- `update_item()`
- online helper methods:
  - `_add_new_user()`
  - `_add_new_item()`
  - `update_model()`

This is the main Python conversion of `MF_fastALS.java`.

#### `src/train.py`
Contains the training loop.

Main functions:
- `run_one_iteration(model)`
- `train_model(model)`

This file mirrors the training loop logic from the Java implementation.

#### `src/predict.py`
Contains the dot-product scoring function used by the model.

#### `src/recommender.py`
Contains:
- `recommend_top_k(model, raw_user_id, k=10)`

This generates top-K recommendations for a user, excluding already-seen training items.

#### `src/split.py`
Offline splitting logic.

Contains:
- `leave_one_out_split(interactions)`

For each user:
- all but the last interaction are training
- the last interaction is test

#### `src/online_split.py`
Online splitting logic.

Contains:
- `chronological_90_10_split(events)`

This implements the online protocol setup:
- first 90% of timestamped events = training
- last 10% = test stream

#### `src/evaluate.py`
Evaluation metrics file.

Contains:
- `hit_rate_at_k()`
- `ndcg_at_k()`
- `online_protocol_metrics()`

The last one implements evaluate-then-update behavior for the online protocol.

#### `src/experiments.py`
Older grid-search style experiment runner.

Useful for broad exploratory testing.

#### `src/sweeps.py`
Main helper file for controlled one-parameter sweeps.

Contains:
- shared experiment-building logic
- offline experiment logic
- online experiment logic
- CSV writing

This file powers all the final sweep runners.

---

## 6. Data folder and what the files mean

### `data/yelp.rating`
This is the main dataset used in the project.

Each line contains:

`user_id item_id rating timestamp`

Meaning:
- `user_id`: user identifier
- `item_id`: item/business identifier
- `rating`: explicit rating value in the raw file
- `timestamp`: Unix timestamp of the interaction

### Important interpretation choice
Although the raw file contains explicit ratings, this project uses it as **implicit feedback**:

- if a `(user, item)` interaction exists, it is treated as observed positive feedback
- the actual rating value is not used in the current model training
- for the offline protocol, only `(user_id, item_id)` pairs are kept
- for the online protocol, timestamps are also kept

This must be explained in the report.

---

## 7. How Spark is used in this implementation

Spark is used in two ways.

### DataFrames
We use Spark DataFrames for:
- reading the raw file
- splitting raw text lines into columns
- selecting columns
- dropping invalid rows
- deduplicating interactions for the offline setting
- grouping by user or item for previews and data understanding

### RDDs
We use Spark RDDs for:
- converting interactions into `(user_id, item_id)` tuples
- grouping user → items
- grouping item → users

### Important limitation
The **training itself is not fully distributed in Spark**.  
The matrix factor updates are executed locally in Python/NumPy after interactions are collected into memory.

Reason:
- the original FastALS algorithm performs custom coordinate-wise updates with dense matrix caches
- for correctness and debugging, a stable local reference implementation was built first
- this is acceptable for a course project, but must be stated as a limitation

---

## 8. Hyperparameters and what they mean

### `factors` (K)
Number of latent factors / embedding dimensions.

Interpretation:
- larger `K` means richer user/item representations
- may improve ranking quality
- also increases runtime and can overfit on small subsets

### `max_iter`
Number of training iterations.

Interpretation:
- more iterations usually reduce loss
- after some point gains may saturate

### `reg`
Regularization strength.

Interpretation:
- prevents very large latent factor values
- helps numerical stability and overfitting control

### `w0`
Global scale of missing-data weight in our implementation.

Interpretation:
- controls how strongly missing entries influence the objective
- larger values make the model care more about unobserved items

### `alpha`
Popularity exponent.

Interpretation:
- determines how much item popularity changes missing-entry weights
- `alpha = 0` means no popularity effect
- larger values emphasize popularity more strongly

### `top_k`
Recommendation cutoff used for HR@K and NDCG@K.

### `w_new`
Weight assigned to a new incoming interaction during online updating.

Interpretation:
- controls how strongly the model reacts to newly observed feedback

### `online_iter`
Number of local online update iterations performed after each new interaction.
In our runs this is usually set to `1`.

---

## 9. Why these baseline values were chosen

We used a broad exploratory search first to identify sensible parameter regions.  
Based on those results, we selected a balanced baseline:

- `factors = 12`
- `max_iter = 5`
- `reg = 0.1`
- `w0 = 1.0`
- `alpha = 0.5`
- `top_k = 10`
- `init_stdev = 0.001`

Reasoning:
- `factors = 12` and `max_iter = 5` gave lower loss than the more minimal settings
- `w0 = 1.0` and `alpha = 0.5` are central, stable values and align with the paper’s interest in popularity-aware weighting
- `init_stdev = 0.001` improved numerical stability

This baseline is not claimed to be globally optimal.  
It is a practical center point for one-parameter sweeps.

---

## 10. Offline protocol used in this project

The offline setting is implemented as follows:

1. load `(user_id, item_id)` interactions
2. remove duplicates
3. for each user:
   - all but the last interaction go to training
   - the last interaction goes to test
4. train the model on the training interactions
5. evaluate recommendations using:
   - HR@10
   - NDCG@10

This is a simplified leave-one-out protocol.

---

## 11. Online protocol used in this project

The online setting is implemented as follows:

1. load `(user_id, item_id, timestamp)` events
2. sort by timestamp
3. split chronologically:
   - first 90% = training
   - last 10% = test stream
4. train the model on the first 90%
5. for each test interaction:
   - generate recommendations and evaluate
   - then insert the new interaction
   - then update only the affected user and item

This is a project-level approximation of the online protocol in the paper.

---

## 12. Important assumptions and deviations from the paper

These must be acknowledged in the report.

### A. Local training instead of distributed Spark training
The algorithm update logic is local NumPy, not fully distributed Spark.

### B. Metrics at 10 instead of 100
We use:
- HR@10
- NDCG@10

The paper reports HR/NDCG at 100.  
Our metrics are therefore stricter and not directly numerically comparable.

### C. Smaller subsets for development and many experiments
Many experiments were run on 5k or 10k interaction subsets for speed and debugging.

### D. Explicit ratings converted into implicit interactions
The rating magnitude is ignored in the current model.
Only existence of interaction is used.

### E. Simplified online update evaluation
Our online protocol is faithful in spirit but still simpler than the full original setup.

### F. No baseline algorithm comparisons
We did not implement BPR, Hu ALS, RCD, etc.

### G. No claim of exact replication
The goal is:
- structural conversion
- meaningful experimental study
- qualitative trend comparison

not exact reproduction of all paper tables or figures.

---

## 13. Why `w_new = 4.0` was chosen

We used `w_new = 4.0` for the online update experiments as a default practical choice.

Reasoning:
- it gives a moderate but nontrivial emphasis to newly arrived interactions
- it is consistent with the general idea that new interactions should have stronger immediate impact in online adaptation
- the original paper also studies the influence of the weight of new interactions in online updating, so using a fixed value here is a controlled simplification

This should be explained as a chosen experimental default, not as an optimized universal value.

---

## 14. What experiments are available

### Exploratory
- `run_experiments.py`
- broad grid-search style experiment

### Final report-style sweeps
- `run_k_sweep.py`
- `run_w0_sweep.py`
- `run_alpha_sweep.py`
- `run_iteration_sweep.py`
- `run_scalability_sweep.py`

Each report-style sweeper saves:
- one CSV for offline
- one CSV for online

---

## 15. What each experiment is meant to show

### `K` sweep
Shows how model complexity affects:
- HR@10
- NDCG@10
- loss
- runtime

### `w0` sweep
Shows how the strength of missing-data weighting affects performance.

### `alpha` sweep
Shows how popularity-aware weighting affects performance.

### iteration sweep
Shows convergence behavior and diminishing returns of more training iterations.

### scalability sweep
Shows how runtime and metrics change as dataset size grows.

This is especially important because the grading rubric explicitly mentions **scalability**.

---

## 16. How to run the project

### Environment
Recommended:
- Python 3.11
- Java 17
- PySpark installed in the virtual environment

### Main run
```bash
python main.py