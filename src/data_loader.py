from typing import List, Tuple
import pandas as pd


def load_interactions_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        nrows=5000
    )
    df = df[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)

    return df


def dataframe_to_interactions(df: pd.DataFrame) -> List[Tuple[int, int]]:
    return list(df[["user_id", "item_id"]].itertuples(index=False, name=None))


def get_unique_users_items(interactions: List[Tuple[int, int]]):
    users = sorted({u for u, _ in interactions})
    items = sorted({i for _, i in interactions})
    return users, items