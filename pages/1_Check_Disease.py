import streamlit as st
import data_1

@st.cache_resource
def load_model():
    return data_1.Data_1()

st.title("Check if you have 'Diabetes' or Not")

glucose = st.number_input("Glucose")
pregnancies = st.number_input("Enter the value of Pregnancies")
bloodPressure = st.number_input("Enter BloodPressure")
skinThickness = st.number_input("Enter the value of SkinThickness")
insulin = st.number_input("Enter the value of Insulin")
diabetesPedigreeFunction = st.number_input("Enter the value of DiabetesPedigreeFunction")
age = st.number_input("Enter Your Age")
weight = st.number_input("Enter your weight (kg)")
height = st.number_input("Enter you height (cm)")   

BMI = weight / ((height/100) ** 2) if height > 0 else 0
st.write(f"Calculated BMI: {BMI:.2f}")

model_object = load_model()

if st.button("Check for Diabetes"):
    try:   
        prediction, probabilities = model_object.predict(
            pregnancies=pregnancies,
            glucose=glucose,
            bloodPressure=bloodPressure,
            skinThickness=skinThickness,     
            insulin=insulin,
            BMI=BMI,
            diabetesPedigreeFunction=diabetesPedigreeFunction,
            age=age
        )
        
        if prediction == 1:
            st.error("🔴 High risk of Diabetes")
        else:
            st.success("🟢 Low risk of Diabetes")
        
        st.write(f"Probability of Diabetes: {probabilities[1]:.2%}")
        st.write(f"Probability of No Diabetes: {probabilities[0]:.2%}")
        
    except Exception as e:
        st.error(f"Error: {e}")