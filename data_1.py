import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

class Data_1:
    
    def __init__(self, pregnancies, glucose, bloodPressure, skinThickness, 
                 insulin, BMI, diabetesPedigreeFunction, age):  
        
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.bloodPressure = bloodPressure
        self.skinThickness = skinThickness
        self.insulin = insulin
        self.BMI = BMI
        self.diabetesPedigreeFunction = diabetesPedigreeFunction
        self.age = age
        
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
    
    def predict(self):
        input_data = pd.DataFrame({
            'Pregnancies': [self.pregnancies],
            'Glucose': [self.glucose],
            'BloodPressure': [self.bloodPressure],
            'SkinThickness': [self.skinThickness],
            'Insulin': [self.insulin],
            'BMI': [self.BMI],
            'DiabetesPedigreeFunction': [self.diabetesPedigreeFunction],
            'Age': [self.age]
        })
        
        prediction = self.xgb_model.predict(input_data)
        prediction_proba = self.xgb_model.predict_proba(input_data)
        
        return prediction[0], prediction_proba[0]