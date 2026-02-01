import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


model_path = Path("model/trained_model.joblib")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()



st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

st.write("Input DF:", input_df)


if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    

    if prediction == 1:
        st.success(f"🎉 Likely to Survive (Probability: {probability:.2f})")
    else:
        st.error(f"⚠️ Likely Not to Survive (Probability: {probability:.2f})")
