import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    # classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
import pickle


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        # "classification_report": classification_report(y_test, y_pred),
    }


def train_and_log_model(name, model, X_train, y_train, X_test, y_test, save_path):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        model_path = f"{save_path}/{name.replace(' ', '_').lower()}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, name.replace(" ", "_").lower())

        print(f"Model {name} trained and logged successfully.")
