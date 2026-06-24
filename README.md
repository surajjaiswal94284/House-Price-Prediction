# House Price Prediction using Machine Learning

A Machine Learning project that predicts California housing prices using a complete preprocessing pipeline and Random Forest Regression.

---

# Features

* Stratified Train-Test Split
* Missing Value Handling using Median Imputation
* Feature Scaling using StandardScaler
* Categorical Encoding using OneHotEncoder
* Feature Pipeline using ColumnTransformer
* Model Selection using Cross Validation
* Random Forest Regression
* Model Persistence using Joblib
* Prediction on New Input Data

---

# Dataset

Dataset used:

```text
housing.csv
```

Target Variable:

```text
median_house_value
```

Features:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity

---

# Project Structure

```text
house-price-prediction-ml/
│
├── housing.csv
├── main.py
├── model_selection.py
├── README.md
├── requirements.txt
└── .gitignore
```

Generated Files (Ignored by Git):

```text
model.pkl
pipeline.pkl
input_data.csv
predictions.csv
```

---

# Machine Learning Workflow

```text
Load Dataset
      ↓
Create Income Categories
      ↓
Stratified Train-Test Split
      ↓
Data Preprocessing
      ↓
Model Selection
      ↓
Train Final Model
      ↓
Save Model & Pipeline
      ↓
Predict on New Data
```

---

# Data Preprocessing

### Numerical Features

* Missing values handled using Median Imputation
* Features scaled using StandardScaler

### Categorical Features

* One-Hot Encoding
* Unknown categories handled safely

### Combined Using

```python
ColumnTransformer
```

---

# Model Selection

The following models were compared using 5-Fold Cross Validation:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

Evaluation Metric:

```text
R² Score
```

Selection Criteria:

* Higher Mean R²
* Lower Standard Deviation

The best performing model was selected as the final model.

---

# Running Model Selection

Execute:

```bash
python model_selection.py
```

This script:

* Preprocesses the dataset
* Compares multiple regression models
* Displays Mean R² Score
* Displays Standard Deviation
* Helps choose the final model

---

# Running Training & Inference

Execute:

```bash
python main.py
```

## First Run

The script:

* Loads dataset
* Creates preprocessing pipeline
* Trains Random Forest Regressor
* Saves:

```text
model.pkl
pipeline.pkl
```

## Future Runs

The script automatically:

* Loads saved model
* Loads saved preprocessing pipeline
* Reads input data
* Generates predictions

---

# Input Data Format

Create a file named:

```text
input_data.csv
```

Required Columns:

```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity
-122.23,37.88,41,880,129,322,126,8.3252,NEAR BAY
```

Run:

```bash
python main.py
```

Predictions will be saved to:

```text
predictions.csv
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib

---

# Concepts Demonstrated

* Data Preprocessing
* Feature Engineering
* Stratified Sampling
* Pipelines
* ColumnTransformer
* One-Hot Encoding
* Feature Scaling
* Cross Validation
* Model Selection
* Random Forest Regression
* Model Persistence
* Inference Pipeline

---

