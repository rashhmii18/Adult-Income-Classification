
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/adult_income_logistic_regression.pkl")

st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="💼",
    layout="wide"
)

st.title("Adult Income Classification")

st.write(
    "Enter the individual's details below to predict whether their income "
    "is <=50K or >50K."
)

st.divider()

st.subheader("Personal and Employment Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=17,
        max_value=90,
        value=30
    )

    workclass = st.selectbox(
        "Workclass",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Masters",
            "9th",
            "Doctorate",
            "5th-6th",
            "Assoc-acdm",
            "Assoc-voc",
            "7th-8th",
            "12th",
            "1st-4th",
            "Prof-school",
            "10th",
            "Preschool"
        ]
    )

    educational_num = st.number_input(
        "Educational Number",
        min_value=1,
        max_value=16,
        value=10
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces"
        ]
    )

with col2:
    relationship = st.selectbox(
        "Relationship",
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried"
        ]
    )

    race = st.selectbox(
        "Race",
        [
            "White",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other",
            "Black"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    native_country = st.selectbox(
        "Native Country",
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands"
        ]
    )

    capital_gain = st.number_input(
        "Capital Gain",
        min_value=0,
        max_value=99999,
        value=0
    )

    capital_loss = st.number_input(
        "Capital Loss",
        min_value=0,
        max_value=4356,
        value=0
    )

    hours_per_week = st.number_input(
        "Hours per Week",
        min_value=1,
        max_value=99,
        value=40
    )

st.divider()

if st.button("Predict Income", use_container_width=True):

    has_capital_gain = int(capital_gain > 0)
    has_capital_loss = int(capital_loss > 0)

    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "education": [education],
        "educational-num": [educational_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country],
        "has_capital_gain": [has_capital_gain],
        "has_capital_loss": [has_capital_loss]
    })

    prediction = model.predict(input_data)[0]
    st.subheader("Prediction")

    if prediction == ">50K":
        st.success("Predicted Income: >50K")
    else:
        st.info("Predicted Income: <=50K")

    st.write("The prediction was generated using the trained Logistic Regression model.")
