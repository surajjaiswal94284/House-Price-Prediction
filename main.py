import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Step 1: Load the dataset
df=pd.read_csv('housing.csv')
print(df.shape)


# Step 2:Stratified Train-Test Split
df['income-cat']=pd.cut(
    df['median_income'],
    bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

strat_train,strat_test=train_test_split(
    df,test_size=0.2,random_state=42,stratify=df['income-cat']
)

strat_train.drop(['income-cat'],axis=1,inplace=True)
strat_test.drop(['income-cat'],axis=1,inplace=True)


# Step 3: Separate features and target variable
X_train=strat_train.drop('median_house_value',axis=1)
Y_train=strat_train['median_house_value'].copy()

X_test=strat_test.drop('median_house_value',axis=1)
Y_test=strat_test['median_house_value'].copy()


# Step 4: Identify numerical and categorical columns
nums_cols=X_train.drop('ocean_proximity',axis=1).columns.tolist()
cat_cols=['ocean_proximity']


# Step 5: Create preprocessing pipelines for numerical and categorical data
num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')), #Handle missing values
    ('scaler',StandardScaler()) # Scale numerical features
])

cat_pipeline=Pipeline([
    ('encoder',OneHotEncoder(handle_unknown='ignore')) 
    # handle_unknown='ignore' is used to avoid errors when the test set contains categories not seen in the training set
])

full_pipeline=ColumnTransformer([
    ('num',num_pipeline,nums_cols),
    ('cat',cat_pipeline,cat_cols)
])

# Step 6: Fit and transform the training data, and transform the test data
X_train_prepared=full_pipeline.fit_transform(X_train) 
X_test_prepared=full_pipeline.transform(X_test)

# print("\nFinal Shapes:")
# print("X_train_prepared:", X_train_prepared.shape)
# print("X_test_prepared:", X_test_prepared.shape)


# Step 7: Train and evaluate models using cross-validation
models={
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Apply cross-validation to each model and print the results
for name,model in models.items():
    scores=cross_val_score(model,X_train_prepared,Y_train,scoring='r2',cv=5)
    print(f"{name}")
    print(f"R2 Scores: {scores}")
    print(f"Mean R2: {scores.mean():.2f}") #R2 mean->1 ke pass(best),0 ke pass(worst))
    print(f"Std Dev: {scores.std():.2f}") #R2 std dev->0 ke pass(best),1 ke pass(worst))
    print()
