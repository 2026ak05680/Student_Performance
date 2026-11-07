# Student Pass/Fail Prediction - ML Assignment 2

M.Tech (AIML), BITS Pilani WILP - Machine Learning

## a. Problem statement

Predict whether a secondary-school student passes the Portuguese language course (final grade G3 >= 10)
using demographic, family, school and lifestyle attributes. Five classification models are implemented on the
same dataset, compared on six evaluation metrics, and demonstrated in an interactive Streamlit app.

## b. Dataset description

- **Source:** UCI Machine Learning Repository - Student Performance dataset (Cortez and Silva, 2008), Portuguese course file `student-por.csv`.
- **Instances:** 649 students. **Input features:** 30 (17 categorical, 13 numeric) - school, sex, age, address, family size, parents' education and jobs, study time, past failures, school/family support, internet access, alcohol use, absences, etc.
- **Target:** `pass_fail` = 1 if final grade `G3` >= 10, else 0. Class balance: 549 pass / 100 fail (about 85% pass).
- **Dropped columns:** `G1` and `G2` (first and second period grades) are removed because they almost reveal `G3`, which would make the task trivial.
- **Split:** stratified 80/20 train/test (519 / 130 rows, random_state = 42). The test rows are saved in `test_data.csv`.
- **Preprocessing:** one-hot encoding for categorical columns and standard scaling for numeric columns, saved inside each model pipeline.

## c. GitHub repository link

<https://github.com/2026ak05680/Student_Performance/tree/main>

Live Streamlit app: <https://studentperformance-35azjh4bxxabjkd8tzftzz.streamlit.app/>

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7923 | 0.6227 | 0.8673 | 0.8909 | 0.8789 | 0.1508 |
| Decision Tree | 0.7692 | 0.6057 | 0.8509 | 0.8818 | 0.8661 | 0.0349 |
| KNN | 0.8154 | 0.5845 | 0.8525 | 0.9455 | 0.8966 | 0.0682 |
| Naive Bayes | 0.6308 | 0.4950 | 0.8690 | 0.6636 | 0.7526 | 0.0857 |
| Support Vector Machine | 0.8308 | 0.6691 | 0.8607 | 0.9545 | 0.9052 | 0.1570 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This Model is Simple and stable baseline. Accuracy is 0.79, with recall 0.89 and F1 0.88, so it finds most of the passing students. AUC (0.62) and MCC (0.15) are low, so it separates pass and fail weakly. It identified only about 5 of the 20 failed students. |
| Decision Tree | Its Overfits the Training data.Accuracy is 0.77 and F1 is 0.87, but MCC is 0.03, almost no better than random guessing. It caught only about 3 of the 20 failing students |
| KNN | Recall is high (0.95) and accuracy is 0.82, but AUC (0.58) and MCC (0.07) are very low. It predicts pass for almost everyone and caught only about 2 of the 20 failing students. The many one-hot columns probably make distances less meaningful|
| Naive Bayes | It has the lowest accuracy (0.63), and its AUC of 0.50 is random level. It catches the most failing students (about 9 of ~20) because it predicts "fail" more often, but it wrongly labels many passing students. The independence assumption is weak because features like Medu/Fedu and Dalc/Walc are correlated |
| Support Vector Machine | It has the best accuracy (0.83), AUC (0.67), recall (0.95), F1 (0.91) and MCC (0.16). Its precision is slightly below Naive Bayes and Logistic Regression. It caught only about 3 of the 20 failing students, and its accuracy is still below the 0.85 "always pass" baseline |
| **Overall winner for this dataset** | SVM is the best on most metrics, but only by a small margin. No model is strong, since every MCC is 0.16 or lower. The reasons are the 85/15 class imbalance and the dropped G1/G2 columns, which were the strongest predictors. With only about 20 fail rows in the test set, small gaps like MCC 0.15 vs 0.16 may be just noise. Class weights, threshold tuning or cross-validation could help. |

## Project structure

```
app.py                  Streamlit app (CSV upload, model dropdown, metrics, confusion matrix, report)
requirements.txt        Dependencies
test_data.csv           Held-out test data (features + pass_fail)
model/                  Saved trained pipelines (.joblib) for all five classifiers
ML_Assignment2_Student_Performance.ipynb   Training and evaluation notebook
```

## How to run locally

```
pip install -r requirements.txt
streamlit run app.py
```
