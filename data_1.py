import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import os

class Data_1:
    
    def __init__(self):  
        model_path = "xgb_model_diabetes.pkl"
        
        if os.path.exists(model_path):
            self.xgb_model = joblib.load(model_path)
        else:
            self.train_and_save_model(model_path)
    
    def train_and_save_model(self, model_path):
        data = pd.read_csv("diabetes.csv")
        target = data["Outcome"]     
        data = data.drop("Outcome", axis="columns")

        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42, stratify=target
        )

        self.xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            random_state=42,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1
        )

        self.xgb_model.fit(train_data, train_target)
        joblib.dump(self.xgb_model, model_path)
    
    def predict(self, pregnancies, glucose, bloodPressure, skinThickness, 
                insulin, BMI, diabetesPedigreeFunction, age):
        
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [bloodPressure],
            'SkinThickness': [skinThickness],
            'Insulin': [insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [diabetesPedigreeFunction],
            'Age': [age]
        })
        
        prediction = self.xgb_model.predict(input_data)
        prediction_proba = self.xgb_model.predict_proba(input_data)
        
        return prediction[0], prediction_proba[0]