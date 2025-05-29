import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'diagnosis' column: M → 1, B → 0
    """
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply StandardScaler to all features (excluding id and label).
    Returns scaled DataFrame with the same structure.
    """
    df = df.copy()
    features = df.drop(columns=["id", "diagnosis"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=features.columns)

    # Reattach id and label
    scaled_df.insert(0, "diagnosis", df["diagnosis"].values)
    scaled_df.insert(0, "id", df["id"].values)
    return scaled_df


def preprocess_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Label encoding
    - Feature scaling
    """
    df = encode_labels(raw_df)
    df = scale_features(df)
    return df


# Feature Selection Methods
def compute_f_classif(X, y):
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)
    return selector.scores_


def compute_mutual_info(X, y):
    selector = SelectKBest(score_func=mutual_info_classif, k="all")
    selector.fit(X, y)
    return selector.scores_


def compute_logistic_l1_importance(X, y):
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    model.fit(X, y)
    return np.abs(model.coef_[0])


def compute_rf_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf.feature_importances_


def perform_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def get_highly_correlated_features(X, threshold=0.9):
    """
    Returns pairs of features with absolute correlation above the given threshold.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (col, row, upper.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if not pd.isnull(upper.loc[row, col]) and upper.loc[row, col] > threshold
    ]

    return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
