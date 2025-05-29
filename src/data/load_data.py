import pandas as pd


def load_raw_data(path: str = "../data/raw/data.csv") -> pd.DataFrame:
    """
    Load raw data from the CSV file.

    Args:
        path (str): Path to the data CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    return df
