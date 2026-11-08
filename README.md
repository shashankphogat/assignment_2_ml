# Breast Cancer Classification — Streamlit App

## a. Problem Statement
Build an end-to-end classification pipeline on the UCI Breast Cancer
Wisconsin (Diagnostic) dataset that predicts whether a tumor is malignant
or benign, using five classifiers, and deploy an interactive Streamlit app.

## b. Dataset Description
- Source: UCI ML Repository — Breast Cancer Wisconsin (Diagnostic)
- Instances: 569 | Features: 30 numeric | Target: `diagnosis` (1 = malignant, 0 = benign)
- Class balance: 357 benign / 212 malignant
- Preprocessing: `id` column dropped; 80/20 stratified train-test split;
  StandardScaler applied inside each model pipeline.

## c. GitHub Repository
https://github.com/shashankphogat/assignment_2_ml

## d. Models Used

### Comparison Table

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree       | 0.9298   | 0.9246 | 0.9048    | 0.9048 | 0.9048 | 0.8492 |
| KNN                 | 0.9561   | 0.9825 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes         | 0.9211   | 0.9891 | 0.9231    | 0.8571 | 0.8889 | 0.8292 |
| SVM                 | 0.9737   | 0.9947 | 1.0000    | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Strong second-best. AUC 0.996, MCC 0.9245, F1 0.9512. Only 3 false negatives — a strong linear baseline on this near-linearly-separable dataset. |
| Decision Tree       | Weakest AUC (0.9246) and lowest MCC (0.8492) — expected, since axis-aligned splits overfit and don't capture feature correlations. |
| KNN                 | Solid (accuracy 0.956, MCC 0.906). After feature scaling KNN performs well, but slightly lower recall (0.9048) shows some boundary sensitivity. |
| Naive Bayes         | Lowest accuracy (0.9211) and lowest recall (0.8571). AUC is surprisingly high (0.9891) — class separation is good, but the independence assumption hurts the threshold-0.5 decision rule. |
| SVM                 | **Best overall.** Highest accuracy (0.9737), AUC (0.9947), F1 (0.9630), and MCC (0.9442). RBF kernel with scaling achieves perfect precision (1.0) — zero false positives. |

**Overall Winner:** Support Vector Machine — highest on all six metrics; recall is 0.9286 (tied with LogReg), everything else SVM leads. The margin over LogReg is small (~0.9 pp accuracy), consistent with this dataset being nearly linearly separable.

## e. Live Streamlit App
https://shashankphogatmlassignment2.streamlit.app

## f. How to Run Locally
```bash
pip install -r requirements.txt
python train_models.py     # regenerates model/*.joblib + test_data.csv
streamlit run app.py
