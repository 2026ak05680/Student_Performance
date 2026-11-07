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

<PASTE YOUR GITHUB REPO LINK HERE>

Live Streamlit app: <PASTE YOUR STREAMLIT APP LINK HERE>

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
| Logistic Regression | WRITE YOUR OWN OBSERVATION |
| Decision Tree | WRITE YOUR OWN OBSERVATION |
| KNN | WRITE YOUR OWN OBSERVATION |
| Naive Bayes | WRITE YOUR OWN OBSERVATION |
| Support Vector Machine | WRITE YOUR OWN OBSERVATION |
| **Overall winner for this dataset** | WRITE YOUR OWN CONCLUSION |

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
