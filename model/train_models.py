"""
Train five classification models for BITS Pilani ML Assignment 2.

Run:
    python train_models.py --data breast_cancer.csv --target diagnosis

This script:
- Splits the original dataset into train/test sets (80/20, stratified).
- Trains five classifiers on the same training set.
- Saves complete preprocessing/model pipelines in the model/ folder.
- Saves test_data.csv (features + actual target) for Streamlit evaluation.
- Saves metrics_comparison.csv for the README/report.
"""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    classes = list(model.classes_)

    # For this assignment dataset there are two classes: B and M.
    # AUC uses the score/probability for the second class in model.classes_.
    if len(classes) == 2:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            scores = None
        auc = roc_auc_score(y_test, scores) if scores is not None else float("nan")
        precision = precision_score(y_test, predictions, pos_label=classes[1], zero_division=0)
        recall = recall_score(y_test, predictions, pos_label=classes[1], zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label=classes[1], zero_division=0)
    else:
        # Weighted averages support multi-class classification too.
        auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": matthews_corrcoef(y_test, predictions),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="breast_cancer.csv", help="Path to input CSV")
    parser.add_argument("--target", default="diagnosis", help="Target column name")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}. Place the CSV beside this script or pass --data.")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available columns: {list(df.columns)}")

    df = df.drop_duplicates().copy()
    y = df[args.target].astype(str)
    X = df.drop(columns=[args.target])

    # This dataset contains numerical input features.
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any():
        raise ValueError("Input features contain missing/non-numeric values. Handle them before training.")
    if y.isnull().any():
        raise ValueError("Target column contains missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for display_name, model in models.items():
        print(f"Training {display_name} ...")
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)
        rows.append({"ML Model Name": display_name, **metrics})

        safe_name = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "K-Nearest Neighbors": "knn",
            "Naive Bayes": "naive_bayes",
            "Support Vector Machine": "svm",
        }[display_name]
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")
        print({k: round(v, 4) for k, v in metrics.items()})

    # Keep test features and true labels together, as required by the assignment.
    test_output = X_test.copy()
    test_output[args.target] = y_test
    test_output.to_csv(BASE_DIR / "test_data.csv", index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(BASE_DIR / "metrics_comparison.csv", index=False)

    metadata = {
        "target_column": args.target,
        "feature_columns": list(X.columns),
        "class_labels": sorted(y.unique().tolist()),
        "test_size": 0.20,
        "random_state": 42,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=4))

    print("\nTraining complete.")
    print(f"Saved five models in: {MODEL_DIR}")
    print(f"Saved test dataset: {BASE_DIR / 'test_data.csv'}")
    print(f"Saved metric comparison: {BASE_DIR / 'metrics_comparison.csv'}")
    print("\nComparison table:")
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    main()
