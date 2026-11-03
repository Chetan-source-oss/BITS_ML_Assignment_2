# BITS Pilani M.Tech (AIML) – Machine Learning Assignment 2
The Github README content

#a. Problem Statement
Implement five machine-learning classification models on one common dataset, evaluate each model using six metrics, and build an interactive Streamlit application. The deployed app loads saved trained models and evaluates uploaded labelled test data without retraining.

#b. Dataset Description
Dataset: Breast Cancer Wisconsin (Diagnostic), accessed through the scikit-learn dataset loader. It is a binary classification dataset.
Property	Description
Instances	569
Input features	30 numerical features
Total columns	31 including target
Target column	Diagnosis
Target labels	M = Malignant; B = Benign
Problem type	Binary classification
Missing values	None in the supplied CSV
The input variables describe cell-nucleus measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry. This is an academic dataset and not a clinical diagnostic tool.
•	Loaded breast_cancer.csv into a pandas DataFrame.
•	Used diagnosis as the target column and the remaining 30 columns as input features.
•	Removed duplicate rows and checked for missing/non-numeric values.
•	Split the data into 80% training and 20% testing using stratify=y and random_state=42.
•	Applied StandardScaler within pipelines for Logistic Regression, KNN and SVM; scaling is fitted on training data only.
Trained all five models using the same training split and evaluated them on the same held-out test split

#c. Github Repository Link
GitHub Repository URL: https://github.com/Chetan-source-oss/BITS_ML_Assignment_2

#d. Models Implemented
Comparison Table with the evaluation metrics calculated for all the 5 models as below:	
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9512	0.9245
Decision Tree	0.9211	0.9448	0.9459	0.8333	0.8861	0.8299
K-Nearest Neighbors	0.9561	0.9823	0.9744	0.9048	0.9383	0.9058
Naive Bayes	0.9386	0.9934	1.0000	0.8333	0.9091	0.8715
Support Vector Machine	0.9737	0.9947	1.0000	0.9286	0.9630	0.9442
observations on the performance of each model on the chosen dataset

ML Model Name	Observation about model performance
Logistic Regression	Accuracy 0.9649, AUC 0.9960, Precision 0.9750, Recall 0.9286, F1 0.9512, MCC 0.9245. Interpret these values together with the confusion matrix and classification report.
Decision Tree	Accuracy 0.9211, AUC 0.9448, Precision 0.9459, Recall 0.8333, F1 0.8861, MCC 0.8299. Interpret these values together with the confusion matrix and classification report.
K-Nearest Neighbors	Accuracy 0.9561, AUC 0.9823, Precision 0.9744, Recall 0.9048, F1 0.9383, MCC 0.9058. Interpret these values together with the confusion matrix and classification report.
Naive Bayes	Accuracy 0.9386, AUC 0.9934, Precision 1.0000, Recall 0.8333, F1 0.9091, MCC 0.8715. Interpret these values together with the confusion matrix and classification report.
Support Vector Machine	Accuracy 0.9737, AUC 0.9947, Precision 1.0000, Recall 0.9286, F1 0.9630, MCC 0.9442. Interpret these values together with the confusion matrix and classification report.
Overall, Winner for your dataset?	In this reproduced run, Support Vector Machine has the highest F1 score (0.9630) and is a candidate overall winner.


