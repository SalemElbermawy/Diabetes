import streamlit as st

st.set_page_config(
    page_title="🩺 Diabetes Risk Project",
    page_icon="💉",
    layout="wide"
)

page_bg = """
<style>
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #6dd5ed);
    background-attachment: fixed;
    color: white;
}

/* Container */
.main-container {
    background: rgba(255, 255, 255, 0.12);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700;
    text-align: center;
}

/* Paragraphs & lists */
p, li {
    color: #f1f1f1 !important;
    font-size: 18px;
    line-height: 1.6;
}

/* Success/info box text */
.stSuccess, .stInfo {
    color: #000 !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown("# 💉 Diabetes Prediction & Risk Assessment")
    st.write("---")

    st.markdown("""
    ## 📖 Project Overview  
    This project is designed to **analyze patient data** and predict their diabetes status.  
    It is divided into **two main parts**:

    1️⃣ **Diagnosis Part**  
    🔍 Determines whether a patient is **diabetic or not** based on medical inputs.  

    2️⃣ **Risk Level Part**  
    ⚠️ If the patient is diabetic, the system predicts the **risk severity**:  
    - 🔵 **Normal**  
    - 🟠 **Moderate**  
    - 🔴 **High / Severe**
    """)

    st.markdown("""
    ## ⚙️ How It Works
    1. 📥 Collect patient inputs (age, lab results, medications, etc.).  
    2. 🤖 Process the data through a **Machine Learning model (XGBoost + Pipelines)**.  
    3. 📊 Output the result:  
       - Whether the patient is **diabetic or not**  
       - If diabetic → **Predict the risk level**  
    """)

    st.success("✨ This project combines AI and Data Science to support doctors in making informed medical decisions.")

    st.markdown('</div>', unsafe_allow_html=True)
