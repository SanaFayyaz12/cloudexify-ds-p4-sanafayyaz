# 🩺 Disease Prediction — Binary Classification

**CloudExify Data Science Internship 2026 — Month 2, Project 4 (FINAL PROJECT)**

**Name:** Sana Fayyaz
**Registration No:** CX-INT-2026-DS-0120

## 📌 Project Description
A machine learning classification project that predicts whether a patient
has a disease (Yes/No) based on medical test results — Age, Blood Pressure,
Glucose, BMI, and Diabetes Pedigree. The notebook covers data exploration,
handling class imbalance, training Logistic Regression and Decision Tree
classifiers, evaluating them with precision/recall/F1/ROC-AUC, and using the
best model to predict a new patient's diagnosis.

## 📁 Files in This Repository
- `disease_prediction.ipynb` — main notebook with all code, outputs, and visualizations
- `disease_data.csv` — dataset of 600 patients (Age, BloodPressure, Glucose, BMI, DiabetesPedigree, Disease)
- `README.md` — this file

## ▶️ How to Run
1. Make sure Python 3 and Jupyter are installed:
   ```
   pip install pandas numpy scikit-learn matplotlib jupyter imbalanced-learn
   ```
2. Open the notebook:
   ```
   jupyter notebook disease_prediction.ipynb
   ```
3. Run all cells from top to bottom (`Cell → Run All`).

## 🧠 Steps Covered in the Notebook
1. Import libraries
2. Load and explore the dataset
3. Handle missing values (filled `BMI` with the median)
4. Check class balance — dataset is imbalanced (75% healthy, 25% disease)
5. Prepare features and target
6. Stratified train/test split (preserves the 75/25 ratio in both sets)
7. Feature scaling with `StandardScaler`
8. Train a Logistic Regression classifier (`class_weight='balanced'`)
9. Train a Decision Tree classifier (`class_weight='balanced'`)
10. Compare both models (Accuracy, Precision, Recall, F1-Score)
11. Confusion matrix
12. ROC curve and AUC score
13. Feature importance analysis
14. Class imbalance handling summary, including a SMOTE oversampling example
15. Predict disease for a new patient
16. Final model report
17. Conclusion

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 80.0% | 56.8% | 83.3% | 0.676 |
| Decision Tree | 75.8% | 51.3% | 66.7% | 0.580 |

**Best Model: Logistic Regression** (highest F1-Score) — chosen because it
best balances precision and recall, which matters most in a medical setting
where missing a real disease case (false negative) is costly.

- **ROC-AUC (Decision Tree):** 0.758
- **Most important feature:** `Glucose`, followed by `BloodPressure` and `BMI`

## ⚖️ Handling Class Imbalance
The dataset had a realistic 75/25 healthy-to-disease imbalance. This was
addressed using:
1. **`class_weight='balanced'`** in both classifiers
2. **Stratified train/test split** to preserve class ratios
3. **SMOTE oversampling** demonstrated as an alternative technique

## 🧠 What I Learned
This project taught me that classification problems — especially in
healthcare — need more than just accuracy to evaluate properly. A model can
have high accuracy simply by predicting "no disease" most of the time, which
is why precision, recall, F1-score, and ROC-AUC matter so much more.
I also learned practical techniques for handling imbalanced datasets, which
is extremely common in real-world medical and fraud-detection data.

## 🔮 What I Would Add With More Time
- Compare additional algorithms (Random Forest, SVM, Gradient Boosting)
- Use cross-validation for more robust performance estimates
- Tune the classification threshold based on the clinical cost of false
  negatives vs. false positives, rather than using the default 0.5 cutoff
