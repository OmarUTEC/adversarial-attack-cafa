import pandas as pd
from typing import Tuple

from src.datasets.preprocess.utils import add_categorical_encoding

_COLS_TO_DROP = ['Timestamp', 'Account', 'Account.1']


def get_amlworld_dataset(
    data_file_path: str,
    metadata_file_path: str,
    encoding_method: str = 'one_hot_encoding'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Preprocess an AMLworld transaction CSV (HI-Small or LI-Small).

    Drops Timestamp and both Account columns (too high cardinality for tabular attacks).
    Keeps: From Bank, To Bank, Amount Received, Amount Paid,
           Receiving Currency, Payment Currency, Payment Format.
    Label: Is Laundering (0 = legitimate, 1 = money laundering).
    """
    df = pd.read_csv(data_file_path)
    metadata_df = pd.read_csv(metadata_file_path)

    df = df.drop(columns=_COLS_TO_DROP, errors='ignore')

    feature_cols = metadata_df[metadata_df.type != 'label'].feature_name.tolist()
    label_col = metadata_df[metadata_df.type == 'label'].feature_name.item()

    df = df[feature_cols + [label_col]].copy()
    df[label_col] = df[label_col].astype(int)

    df, metadata_df = add_categorical_encoding(df, metadata_df, encoding_method=encoding_method)

    x_df, y_df = df.drop(columns=[label_col]), df[label_col]
    return x_df, y_df, metadata_df
