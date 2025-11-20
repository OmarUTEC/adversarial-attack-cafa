import pandas as pd
from typing import Tuple, List

from src.datasets.preprocess.utils import add_categorical_encoding


def get_creditcard_dataset(
    data_file_path: str,
    metadata_file_path: str,
    encoding_method: str = 'one_hot_encoding'
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Preprocess the credit card transactions dataset.
    The data is already numeric; we simply align the column order using the metadata
    and optionally apply categorical encoding (no-op here because there are no categorical columns).
    """
    df = pd.read_csv(data_file_path)
    metadata_df = pd.read_csv(metadata_file_path)

    # Ensure deterministic column order and filter down to expected schema
    df = df[metadata_df['feature_name']].copy()

    label_col = metadata_df[metadata_df.type == 'label'].feature_name.item()  # 'Class'
    df[label_col] = df[label_col].astype(int)

    # Apply categorical encoding if requested (kept for API compatibility)
    df, metadata_df = add_categorical_encoding(df, metadata_df, encoding_method=encoding_method)

    x_df, y_df = df.drop(columns=[label_col]), df[label_col]

    return x_df, y_df, metadata_df
