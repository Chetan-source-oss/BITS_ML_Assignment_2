# BITS Pilani M.Tech (AIML) – Machine Learning Assignment 2

## Problem statement
Build and evaluate five classification models on one dataset, then provide an interactive Streamlit application that evaluates uploaded labelled test data using saved models.

## Dataset description
Breast Cancer Wisconsin (Diagnostic) dataset. It contains 569 instances, 30 numerical input features and one target column named `diagnosis`. Target labels are `B` (Benign) and `M` (Malignant). The dataset is used for educational classification work and is not a clinical diagnostic tool.

## Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

## Evaluation metrics
Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

After running `train_models.py`, copy the actual results from `metrics_comparison.csv` into the comparison table below. Do not invent or estimate metric values.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| K-Nearest Neighbors | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

## Observations
Write observations based on your actual metric values and confusion matrices. Discuss which model has the best overall balance and why. The overall winner should be selected from the measured results, not assumed in advance.

## GitHub Repository Link
Add your GitHub repository URL here.

## Live Streamlit App Link
Add your deployed Streamlit app URL here.

## How to run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Put `breast_cancer.csv` in this project folder.
3. Train and save the models: `python train_models.py --data breast_cancer.csv --target diagnosis`
4. Start the app: `streamlit run app.py`

The training script creates the `model/` directory contents, `test_data.csv`, and `metrics_comparison.csv`.
