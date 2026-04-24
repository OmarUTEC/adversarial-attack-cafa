from typing import Dict, Tuple

import pandas as pd

from src.datasets.preprocess.utils import add_categorical_encoding


def _read_li_small_file(data_file_path: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(data_file_path)
    return df[metadata_df["feature_name"]].copy()


def _coerce_types(df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    label_col = metadata_df[metadata_df.type == "label"].feature_name.item()
    ordinal_features = metadata_df[metadata_df.type == "ordinal"].feature_name.tolist()
    continuous_features = metadata_df[metadata_df.type == "continuous"].feature_name.tolist()

    for feature_name in ordinal_features:
        df[feature_name] = df[feature_name].astype(int)
    for feature_name in continuous_features:
        df[feature_name] = df[feature_name].astype(float)
    df[label_col] = df[label_col].astype(int)
    return df


def _split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[label_col]), df[label_col]


def get_li_small_dataset(
    data_file_path: str,
    metadata_file_path: str,
    encoding_method: str = None,
    val_file_path: str = None,
    test_file_path: str = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame] | Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Preprocess LI-Small AML transactions.

    If validation and test paths are provided, returns predefined train/val/test splits.
    Otherwise, returns a single processed dataset compatible with the legacy random split path.
    """
    metadata_df = pd.read_csv(metadata_file_path)
    label_col = metadata_df[metadata_df.type == "label"].feature_name.item()

    train_df = _coerce_types(_read_li_small_file(data_file_path, metadata_df), metadata_df)

    if val_file_path and test_file_path:
        val_df = _coerce_types(_read_li_small_file(val_file_path, metadata_df), metadata_df)
        test_df = _coerce_types(_read_li_small_file(test_file_path, metadata_df), metadata_df)
        combined_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
        combined_df, encoded_metadata_df = add_categorical_encoding(
            combined_df,
            metadata_df,
            encoding_method=encoding_method,
        )

        train_end = len(train_df)
        val_end = train_end + len(val_df)
        train_encoded = combined_df.iloc[:train_end].reset_index(drop=True)
        val_encoded = combined_df.iloc[train_end:val_end].reset_index(drop=True)
        test_encoded = combined_df.iloc[val_end:].reset_index(drop=True)

        return {
            "train": _split_xy(train_encoded, label_col),
            "val": _split_xy(val_encoded, label_col),
            "test": _split_xy(test_encoded, label_col),
            "metadata_df": encoded_metadata_df,
        }

    train_df, metadata_df = add_categorical_encoding(
        train_df,
        metadata_df,
        encoding_method=encoding_method,
    )
    x_df, y_df = _split_xy(train_df, label_col)
    return x_df, y_df, metadata_df
