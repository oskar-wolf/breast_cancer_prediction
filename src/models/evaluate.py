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


from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns


def plot_all_confusion_matrices(models_dict, X_test, y_test):
    """
    Plot confusion matrices for all models in a single figure.
    """
    num_models = len(models_dict)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
        axes[idx].set_title(f"Confusion Matrix - {name}")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_all_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for all models on a single plot.
    """
    plt.figure(figsize=(10, 8))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            print(f"Skipping {name} – no probability/decision output available.")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
