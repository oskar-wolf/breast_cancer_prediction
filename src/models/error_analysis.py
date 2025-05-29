import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_misclassified_indices(y_true, y_pred):
    """Return indices of False Positives and False Negatives."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fp = np.where((y_pred == 1) & (y_true == 0))[0]
    fn = np.where((y_pred == 0) & (y_true == 1))[0]
    return fp, fn


def inspect_misclassified_samples(
    X_test, y_true, y_pred, fp_indices, fn_indices, max_rows=5
):
    """Return a DataFrame showing misclassified samples for manual inspection."""
    fp_df = pd.DataFrame(X_test.iloc[fp_indices])
    fp_df["True Label"] = y_true.iloc[fp_indices].values
    fp_df["Predicted"] = y_pred[fp_indices]

    fn_df = pd.DataFrame(X_test.iloc[fn_indices])
    fn_df["True Label"] = y_true.iloc[fn_indices].values
    fn_df["Predicted"] = y_pred[fn_indices]

    print("\n🔴 False Positives (Predicted Malignant, Actually Benign):")
    display(fp_df.head(max_rows))

    print("\n🔵 False Negatives (Predicted Benign, Actually Malignant):")
    display(fn_df.head(max_rows))
