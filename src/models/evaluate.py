import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import pandas as pd


def load_models(models_dir="../models"):
    models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith(".pkl"):
            path = os.path.join(models_dir, filename)
            with open(path, "rb") as f:
                model_name = filename.replace(".pkl", "")
                models[model_name] = pickle.load(f)
    return models


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }

    if y_proba is not None:
        metrics["AUC"] = roc_auc_score(y_test, y_proba)

    return (
        metrics,
        confusion_matrix(y_test, y_pred),
        roc_curve(y_test, y_proba) if y_proba is not None else None,
    )
