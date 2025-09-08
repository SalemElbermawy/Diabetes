
import streamlit as st
from data_2 import Data

@st.cache_resource
def load_model():
    return Data()

st.title("🔬 Diabetes A1C Result Prediction")



race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic'])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.selectbox("Age", [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
])
time_in_hospital = st.number_input("Time in hospital (days)", min_value=1)
num_lab_procedures = st.number_input("Number of lab procedures", min_value=1)
num_procedures = st.number_input("Number of procedures", min_value=0)
num_medications = st.number_input("Number of medications", min_value=0)
number_outpatient = st.number_input("Number of outpatient visits", min_value=0)
number_emergency = st.number_input("Number of emergency visits", min_value=0)
number_inpatient = st.number_input("Number of inpatient visits", min_value=0)
number_diagnoses = st.number_input("Number of diagnoses", min_value=0)

max_glu_serum = st.selectbox("Max glucose serum", [None, '>300', 'Norm', '>200'])
metformin = st.selectbox("Metformin", ['No', 'Steady', 'Up', 'Down'])
repaglinide = st.selectbox("Repaglinide", ['No', 'Up', 'Steady', 'Down'])
nateglinide = st.selectbox("Nateglinide", ['No', 'Steady', 'Down', 'Up'])
chlorpropamide = st.selectbox("Chlorpropamide", ['No', 'Steady', 'Down', 'Up'])
glimepiride = st.selectbox("Glimepiride", ['No', 'Steady', 'Down', 'Up'])
acetohexamide = st.selectbox("Acetohexamide", ['No', 'Steady'])
glipizide = st.selectbox("Glipizide", ['No', 'Steady', 'Up', 'Down'])
glyburide = st.selectbox("Glyburide", ['No', 'Steady', 'Up', 'Down'])
tolbutamide = st.selectbox("Tolbutamide", ['No', 'Steady'])
pioglitazone = st.selectbox("Pioglitazone", ['No', 'Steady', 'Up', 'Down'])
rosiglitazone = st.selectbox("Rosiglitazone", ['No', 'Steady', 'Up', 'Down'])
acarbose = st.selectbox("Acarbose", ['No', 'Steady', 'Up', 'Down'])
miglitol = st.selectbox("Miglitol", ['No', 'Steady', 'Down', 'Up'])
troglitazone = st.selectbox("Troglitazone", ['No', 'Steady'])
tolazamide = st.selectbox("Tolazamide", ['No', 'Steady', 'Up'])

examide = "No"
citoglipton = "No"

insulin = st.selectbox("Insulin", ['No', 'Up', 'Steady', 'Down'])
glyburide_metformin = st.selectbox("Glyburide-metformin", ['No', 'Steady', 'Down', 'Up'])
glipizide_metformin = st.selectbox("Glipizide-metformin", ['No', 'Steady'])
glimepiride_pioglitazone = st.selectbox("Glimepiride-pioglitazone", ['No', 'Steady'])
metformin_rosiglitazone = st.selectbox("Metformin-rosiglitazone", ['No', 'Steady'])
metformin_pioglitazone = st.selectbox("Metformin-pioglitazone", ['No', 'Steady'])

change = st.selectbox("Change", ['No', 'Ch'])
diabetesMed = st.selectbox("DiabetesMed", ['No', 'Yes'])
readmitted = st.selectbox("Readmitted", ['NO', '>30', '<30'])
# تحميل النموذج مسبقاً
model_object = load_model()

if st.button("🔍 Predict A1C Result"):
    try:
        prediction = model_object.predict(
            race, gender, age, time_in_hospital, num_lab_procedures,
            num_procedures, num_medications, number_outpatient,
            number_emergency, number_inpatient, number_diagnoses,
            max_glu_serum, metformin, repaglinide, nateglinide,
            chlorpropamide, glimepiride, acetohexamide, glipizide,
            glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose,
            miglitol, troglitazone, tolazamide, examide, citoglipton,
            insulin, glyburide_metformin, glipizide_metformin,
            glimepiride_pioglitazone, metformin_rosiglitazone,
            metformin_pioglitazone, change, diabetesMed, readmitted
        )

        if prediction < 6:
            st.success(f"🟢 Prediction: {prediction:.2f} → Normal")
        elif 6 <= prediction < 8:
            st.warning(f"🟠 Prediction: {prediction:.2f} → Medium Risk")
        else:
            st.error(f"🔴 Prediction: {prediction:.2f} → High Risk")

    except Exception as e:
        st.error(f"Error: {e}")
