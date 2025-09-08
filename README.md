# Diabetes Risk Prediction App 🩺

## Overview
This is a **Diabetes Risk Prediction Application** built using **Python** and **Streamlit**.  
The project aims to help users **check if they are at risk of diabetes** and, if so, **estimate the severity or risk level** based on their medical and lifestyle inputs.

The app is divided into **two main parts**:

1. **Diabetes Diagnosis** 🧪  
   - Users enter personal and medical details like glucose levels, blood pressure, BMI, insulin, age, and other relevant features.  
   - The app predicts whether the user has a high or low risk of diabetes.  
   - Probabilities for each outcome are shown for clarity.

2. **Diabetes Severity / Risk Duration Estimation** ⏳  
   - For users predicted at risk, the app estimates the potential **risk level** or **time severity**, helping users understand their condition better.  
   - This part uses advanced features from patient records, medications, and hospital visit history.

---

## Features
- ✅ User-friendly **Streamlit interface**  
- ✅ Calculates **BMI automatically** from weight and height  
- ✅ Uses **XGBoost machine learning models** for prediction  
- ✅ Displays both **risk classification** and **probability scores**  
- ✅ Handles categorical and numerical inputs with proper preprocessing  

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-risk-app.git
cd diabetes-risk-app
