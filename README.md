# IML Challenge 2 — Russian Cities Housing Price Prediction

This repository contains **my solution** for IML Challenge 2, where I predicted housing prices across Russian cities using **Random Forest Regression** with custom preprocessing and feature engineering.

I ranked **40th out of 139 teams** with an **RMSLE score of 12.49078**.



## My Approach

### 1. Data Cleaning

The dataset was messy and required extensive cleaning. I handled:

* Missing values in both numeric and categorical columns
* Inconsistent formatting
* Noisy fields, especially location-based attributes
* Outliers in numeric features

### 2. Preprocessing & Feature Engineering

I built a structured preprocessing pipeline using **scikit-learn** to transform the data before training.

#### Numerical Features

* Imputed missing values using **median**
* Scaled values using **StandardScaler**

```python
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

#### Categorical Features

* Filled missing values with **most frequent category**
* Encoded categories using **OrdinalEncoder**
* Handled unseen categories safely using `unknown_value = -1`

```python
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
```

#### Final Combined Pipeline

```python
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])
```

This pipeline ensured the model received **clean, consistent, and well-transformed features**.

---

### 3. Modeling

I experimented with multiple models but found **Random Forest Regressor** performed the best due to:

* Resistance to noise
* Ability to handle nonlinear relationships
* Stability across messy real-world data

Key hyperparameters tuned:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `max_features`

**Final Score:** `12.49078 RMSLE`
**Leaderboard Rank:** `40 / 139`

<img width="1920" height="526" alt="image" src="https://github.com/user-attachments/assets/33c9aed1-be70-44ac-99a0-c0ffab4473b7" />

---

## Repository Contents

```
notebooks/
│── EDA.ipynb                # Data exploration and analysis
│── Model_Training.ipynb     # Preprocessing pipeline + Random Forest model
│── Submission_task2.csv     # Final predictions for leaderboard
```

> All work is contained inside **Jupyter notebooks**.

---

## Results Summary

* **Competition:** IML Challenge 2
* **Task:** Predict housing prices for Russian cities
* **Rank:** Top 30% (40 / 139)
* **Score:** 12.49078 RMSLE
* **Model Used:** Random Forest Regressor

---

## What This Project Demonstrates

* Working with **messy, real-world data**
* Proper use of **scikit-learn pipelines**
* **Feature engineering** for structured datasets
* **Model selection and tuning**
* End-to-end **machine learning workflow**

