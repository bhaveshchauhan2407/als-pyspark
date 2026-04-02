import numpy as np


def predict_score(user_vector: np.ndarray, item_vector: np.ndarray) -> float:

    return float(np.dot(user_vector, item_vector))