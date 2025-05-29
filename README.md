
# 🧠 Breast Cancer Prediction with Interpretable ML

This project focuses on developing machine learning models to predict whether a tumor is **benign** or **malignant** based on diagnostic features from the Breast Cancer Wisconsin dataset. The goal is not just high accuracy, but also **interpretability** and **transparency** in the decision-making process.

---

## 📁 Project Structure

```
├── data/                  # Raw, interim, and processed datasets
│   ├── raw/               # Original input data
│   ├── interim/           # Intermediate transformation files (e.g., PCA, splits)
│   └── processed/         # Final processed feature and split files
├── models/                # Trained model files (.pkl)
├── notebooks/             # Jupyter notebooks for each stage
├── src/                   # Source code: data, model, evaluation, interpretability
├── mlruns/                # MLflow experiment tracking folder
├── environment.yml        # Conda environment file
├── LICENSE
```

---

## ✅ Objectives

- Build interpretable and accurate classification models.
- Compare performance using **full feature set** vs. **top 10 selected features**.
- Visualize and analyze model decisions with tools like **SHAP** and **EBM**.
- Track experiments using **MLflow**.
- Perform **error analysis** to identify model limitations.

---

## 🧪 Models Trained

- Logistic Regression
- Decision Tree
- Random Forest
- Explainable Boosting Machine (EBM)
- XGBoost

Each model was trained on:
- **Full feature set**
- **Top 10 selected features** (based on ANOVA, mutual information, RF, Logistic L1)

---

## 📊 Evaluation Metrics

We evaluated the models on:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix
- ROC Curve

> Best models were selected with F1 Score ≥ 0.95

---

## 📈 Interpretability

- **Global Interpretability**:
  - Logistic Regression coefficients
  - SHAP summary plots (bar & beeswarm)
  - EBM global term importance

- **Local Interpretability**:
  - SHAP waterfall plots (individual predictions)
  - EBM local explanations

- **Feature Correlation**:
  - Highly correlated features (correlation > 0.9) were identified and handled to avoid redundancy.

---

## 🔧 Tools & Libraries

- `scikit-learn`
- `xgboost`
- `interpret`
- `shap`
- `mlflow`
- `matplotlib`, `seaborn`
- `pandas`, `numpy`

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-repo/breast-cancer-prediction.git
cd breast-cancer-prediction

# Create and activate Conda environment
conda env create -f environment.yml
conda activate bc38

# Launch MLflow UI
mlflow ui
```

---

## 📬 Notes

- The project includes all stages: data prep, training, evaluation, and interpretability.
- A detailed report documenting methodology, findings, and challenges will be provided separately.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
