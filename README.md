# California Housing Price Prediction

End-to-end ML pipeline to predict housing prices using the California Housing Dataset.

## Features
- Stratified Sampling
- Missing Value Imputation (SimpleImputer)
- Categorical Encoding (OneHotEncoder)
- Feature Scaling (StandardScaler)
- Pipeline + ColumnTransformer
- Cross Validation (Linear Regression, Decision Tree, Random Forest comparison)
- Model Persistence (joblib)

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn

## How to Run
python main.py

## Project Structure
 housing.csv
 main.py
 model.pkl (generated after training)
 pipeline.pkl (generated after training)