import os
import joblib  #used for loading the model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor



MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

def build_pipeline(num_col,cat_col):
    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='median')), #Handle missing values
        ('scaler',StandardScaler()) # Scale numerical features
    ])

    cat_pipeline=Pipeline([
        ('encoder',OneHotEncoder(handle_unknown='ignore')) 
        # handle_unknown='ignore' is used to avoid errors when the test set contains categories not seen in the training set
    ])

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_col),
        ('cat',cat_pipeline,cat_col)
    ])
    
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    #TRAINING PHASE - First time training the model
    # Step 1: Load the dataset
    df=pd.read_csv('housing.csv')
    print(df.shape)


    # Step 2:Stratified Train-Test Split
    df['income_cat']=pd.cut(
        df['median_income'],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    strat_train,strat_test=train_test_split(
        df,test_size=0.2,random_state=42,stratify=df['income_cat']
    )

    strat_train.drop(['income_cat'],axis=1,inplace=True)
    strat_test.drop(['income_cat'],axis=1,inplace=True)


    # Step 3: Separate features and target variable
    X_train=strat_train.drop('median_house_value',axis=1)
    Y_train=strat_train['median_house_value'].copy()

    X_test=strat_test.drop('median_house_value',axis=1)
    Y_test=strat_test['median_house_value'].copy()


    # Step 4: Identify numerical and categorical columns
    nums_cols=X_train.drop('ocean_proximity',axis=1).columns.tolist()
    cat_cols=['ocean_proximity']

    
    full_pipeline=build_pipeline(nums_cols,cat_cols)

    
    X_train_prepared=full_pipeline.fit_transform(X_train)
    
    model=RandomForestRegressor(random_state=42)
    
    model.fit(X_train_prepared,Y_train)

    
    joblib.dump(model,MODEL_FILE)
    
    joblib.dump(full_pipeline,PIPELINE_FILE)
else:
    # INFERENCE PHASE - Model is already ready just to use it
    model=joblib.load(MODEL_FILE)
    full_pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv('input_data.csv')
    
    input_data_prepared=full_pipeline.transform(input_data) #Test data should be transformed using the same pipeline as the training data
    
    predict=model.predict(input_data_prepared)
    
    input_data['predicted_median_house_value']=predict # save the predictions in the input data dataframe
    
    input_data.to_csv('predictions.csv',index=False) # save the predictions to a new csv file
