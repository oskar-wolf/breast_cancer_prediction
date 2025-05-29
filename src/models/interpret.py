import shap
import matplotlib.pyplot as plt
import numpy
import pandas as pd


def explain_logistic_coefficients(model, feature_names):
    """
    Plot the coefficients of a logistic regression model.
    """
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})

    coef_df = coef_df.sort_values(by="Coefficient", ascending=False, key=abs)

    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="skyblue")
    plt.xlabel("Coefficient Value")
    plt.title("Logistic Regression Coefficients")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_ebm_importance(ebm_model):
    """plot the feature importance of an Explainable Boosting Machine (EBM) model."""
    ebm_model_global = ebm_model.explain_global()
    from interpret import show

    show(ebm_model_global)


# def shap_summary_plot(model, X_train):
#     """create a SHAP summary plot for the model."""
#     explainer = shap.Explainer(model, X_train)
#     shap_values = explainer(X_train)

#     # Bar plot of feature importance
#     shap.summary_plot(
#         shap_values, X_train, plot_type="bar", feature_names=X_train.columns
#     )

#     # Full SHAP beeswarm plot
#     shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)


def shap_summary_plot(model, X_train):
    """
    Create SHAP summary plots (bar + beeswarm) for tree-based models like RF, XGB.
    """
    # Force TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # If classifier: shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 = malignant

    # Bar plot of feature importance
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # Beeswarm plot
    shap.summary_plot(shap_values, X_train)


def shap_local_explanation(model, X_train, instance_index=0):
    """
    Create a SHAP local explanation (waterfall plot) for a specific instance.
    """
    # Force TreeExplainer for tree models to avoid fallback behavior
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # If it's a list (per-class), choose class 1 (malignant)
    if isinstance(shap_values, list):
        shap_value = shap_values[1][instance_index]
        base_value = explainer.expected_value[1]
    else:
        shap_value = shap_values[instance_index]
        base_value = explainer.expected_value

    # Waterfall plot for local explanation
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_value,
            base_values=base_value,
            data=X_train.iloc[instance_index],
            feature_names=X_train.columns,
        )
    )
