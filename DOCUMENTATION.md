# 📊 Project 4 Documentation — Disease Prediction (Classification)

**CloudExify Data Science Internship 2026 — Month 2 (FINAL PROJECT)**

**Name:** Sana Fayyaz
**Registration No:** CX-INT-2026-DS-0120

---

## 1. Project Overview

This project builds a binary classification model that predicts whether a
patient has a disease (Yes/No) based on medical test results — Age, Blood
Pressure, Glucose, BMI, and Diabetes Pedigree. It is the final and most
advanced project of the internship, introducing classification metrics and
techniques for handling imbalanced medical data.

## 2. Objective

To build a working classification pipeline that:
- Loads and explores a medical dataset
- Checks and addresses class imbalance
- Trains and compares multiple classification algorithms
- Evaluates performance using classification-specific metrics
- Predicts disease presence for a new, unseen patient

## 3. Data Science / ML Concepts Used

| Concept | Where Used |
|---|---|
| Binary classification | Predicting Disease = 0 or 1 |
| Class imbalance detection | `value_counts()` on the target column |
| Stratified train/test split | `train_test_split(..., stratify=y)` |
| Feature scaling | `StandardScaler` for Logistic Regression |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` |
| Decision Tree Classifier | `sklearn.tree.DecisionTreeClassifier` |
| Classification metrics | Accuracy, Precision, Recall, F1-Score |
| Confusion Matrix | `confusion_matrix`, `ConfusionMatrixDisplay` |
| ROC Curve & AUC | `roc_curve`, `auc` |
| Feature importance | Extracted from the Decision Tree |
| Class balancing techniques | `class_weight='balanced'`, SMOTE oversampling |

## 4. Dataset

`disease_data.csv` — 600 patient records with columns: `Age`,
`BloodPressure`, `Glucose`, `BMI`, `DiabetesPedigree`, and `Disease`
(0 = No, 1 = Yes). The dataset has a realistic class imbalance — 450 healthy
patients (75%) and 150 disease cases (25%) — with a few missing `BMI`
values introduced for realism.

## 5. Steps Performed (Notebook Walkthrough)

1. Import libraries
2. Load and explore the dataset
3. Handle missing values — filled `BMI` with the median
4. Checked class balance (found 75/25 imbalance) and visualized it
5. Prepared features (X) and target (y)
6. Stratified train/test split to preserve class ratios in both sets
7. Scaled features using `StandardScaler`
8. Trained a Logistic Regression classifier with `class_weight='balanced'`
9. Trained a Decision Tree classifier with `class_weight='balanced'`
10. Compared both models using Accuracy, Precision, Recall, and F1-Score
11. Generated a confusion matrix for the Decision Tree
12. Plotted the ROC curve and calculated the AUC score
13. Analyzed feature importance from the Decision Tree
14. Demonstrated SMOTE oversampling as an alternative imbalance-handling technique
15. Predicted disease presence for a new patient
16. Generated a final summary report

## 6. Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 80.0% | 56.8% | 83.3% | 0.676 |
| Decision Tree | 75.8% | 51.3% | 66.7% | 0.580 |

**Best Model: Logistic Regression** — selected based on F1-Score, which
balances precision and recall. In a medical context, Recall (catching
actual disease cases) is especially important, and Logistic Regression
scored highest here at 83.3%.

- **ROC-AUC (Decision Tree):** 0.758
- **Most important feature:** `Glucose` (0.466), followed by `BloodPressure` (0.182) and `BMI` (0.159)

## 7. Testing Checklist

| Test Case | Result |
|---|---|
| Load disease dataset | ✅ Data loaded, classes shown (450 / 150) |
| Check class balance | ✅ Imbalance detected (75%/25%) |
| Train/test split | ✅ Correct stratified 80/20 split |
| Logistic Regression accuracy | ✅ 0.800 (> 0.8 target, on the edge) |
| Decision Tree vs Logistic | Logistic Regression outperformed on F1/Recall |
| Confusion matrix | ✅ TP/TN/FP/FN clearly shown |
| ROC curve | AUC = 0.758 (slightly below the 0.8 target, noted below) |
| Feature importance | ✅ Top features identified |
| Predict new patient | ✅ Prediction + probability returned |

## 8. Challenges Faced & Solutions

- **Class imbalance in the target variable:** Solved using a combination of
  `class_weight='balanced'` in both models and a stratified train/test
  split, so the minority (disease) class wasn't ignored during training.
- **Choosing the right evaluation metric:** Accuracy alone would have been
  misleading here, since a model that always predicts "healthy" would still
  score ~75% accuracy. Solved by prioritizing F1-Score and Recall instead.
- **Understanding the AUC result:** The Decision Tree's AUC (0.758) came in
  slightly below the 0.8 benchmark suggested in the guide. This is a
  realistic outcome on a modestly-sized, noisy synthetic dataset, and is
  noted transparently in the report rather than adjusted artificially.

## 9. What I Learned

This project taught me that classification success can't be judged by
accuracy alone — especially in healthcare, where missing an actual disease
case (a false negative) is far more costly than a false alarm. I learned to
read a confusion matrix, interpret an ROC curve, and apply practical
techniques (class weighting, stratified splits, SMOTE) for handling
imbalanced real-world data.

## 10. What I Would Add With More Time

- Compare additional algorithms such as Random Forest, SVM, and Gradient Boosting
- Use k-fold cross-validation for more robust, less split-dependent metrics
- Tune the classification decision threshold based on the real-world cost
  of false negatives vs. false positives, rather than using the default 0.5

## 11. Files Submitted

- `disease_prediction.ipynb`
- `disease_data.csv`
- `README.md`
- `DOCUMENTATION.md` (this file)
