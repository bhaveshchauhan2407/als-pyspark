# ALS-PySpark: Fast Matrix Factorization for Online Recommendation

## Overview
This project is a PySpark implementation of the **Fast Matrix Factorization for Online Recommendation with Implicit Feedback** algorithm, based on the paper by He et al. (SIGIR 2016). It was developed as part of our Database Management course APM_50443_EP.

**Contributors:** Bhavesh CHAUHAN & Korouhanba Khuman LAIKHURAM

---

## Background
Traditional recommendation systems rely on explicit feedback such as star ratings. In reality, most user data is implicit — purchases, views, clicks — and lacks clear negative signals. This project tackles that challenge by implementing the eALS (element-wise Alternating Least Squares) algorithm, which learns user and item representations from implicit feedback efficiently and supports real-time model updates.

The core idea is that instead of treating all unobserved interactions equally, the algorithm weights missing data based on item popularity — popular items a user ignored are more likely to be true negatives.

---

## What This Project Implements

- **Matrix Factorization on Implicit Feedback** — models user-item interactions without explicit ratings
- **Popularity-aware Weighting** — assigns higher weights to popular unobserved items as likely negatives
- **Fast eALS Learning Algorithm** — element-wise coordinate descent that avoids expensive matrix inversions, running K times faster than conventional ALS
- **Online Incremental Learning** — instantly refreshes the model as new user interactions stream in, without full retraining
- **PySpark Implementation** — leverages RDD and DataFrame APIs for parallelized, large-scale computation

---

## Algorithm: eALS

The eALS algorithm optimizes matrix factorization one parameter at a time using memoization to avoid redundant computation over the full missing data space.

| Method | Time Complexity |
|---|---|
| Conventional ALS | O((M+N)K³ + \|R\|K²) |
| eALS (this project) | O((M+N)K² + \|R\|K) |

This makes eALS K times faster than ALS while achieving better accuracy through non-uniform weighting. The algorithm is also parallel, making it well-suited for Spark-based distributed execution.

---

## Datasets

Experiments were conducted on yelp ratings dataset, preprocessed to remove users and items with fewer than 10 interactions:

| Dataset | Reviews | Users | Items | Sparsity |
|---|---|---|---|---|
| Yelp | 731,671 | 25,677 | 25,815 | 99.89% |

---

## Evaluation

The implementation was evaluated under two protocols:
- **Offline Protocol** — leave-one-out evaluation on static historical data
- **Online Protocol** — simulated data stream using chronological split (90% train / 10% test)

Metrics used: **Hit Ratio (HR)** and **Normalized Discounted Cumulative Gain (NDCG)** at top-10 & top-100.

---

## Reference
> Xiangnan He, Hanwang Zhang, Min-Yen Kan, Tat-Seng Chua. *Fast Matrix Factorization for Online Recommendation with Implicit Feedback*. SIGIR 2016. DOI: 10.1145/2911451.2911489
[link](https://github.com/hexiangnan/sigir16-eals)  