# ClimateGuard Dashboard (Optimized with Caching)
# Author: Monica Yuol Manyok

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit settings
st.set_page_config(page_title="ClimateGuard", page_icon="🌍", layout="centered")
sns.set_style("whitegrid")

# -----------------------
# Cached Functions
# -----------------------
@st.cache_data
def load_flood_data():
    return pd.read_excel("Pluvial_Flood_Dataset.xlsx")

@st.cache_data
def load_fire_data():
    return pd.read_csv("California_Fire_Incidents.csv")

@st.cache_resource
def train_flood_model(df):
    X = df[["Slope", "Curvature ", "Aspect", "TWI", "FA", "Drainage", "Rainfall"]]
    y = df["SUSCEP"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X, y, X_test, y_test

@st.cache_resource
def train_fire_model(df):
    X = df[["AcresBurned", "Latitude", "Longitude", "Fatalities", "Injuries"]].fillna(0)
    y = df["Status"].fillna("Unknown")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X, y, X_test, y_test

# -----------------------
# App Header
# -----------------------
st.title("🌍 ClimateGuard - AI Disaster Prediction")
st.write("Built by **Monica Yuol Manyok**")

# Sidebar
option = st.sidebar.selectbox(
    "Choose a Disaster Model:",
    ("🏠 Home", "Flood Prediction 🌊", "Wildfire Prediction 🔥")
)

# -----------------------
# Landing Page
# -----------------------
if option == "🏠 Home":
    st.header("Welcome to ClimateGuard 🚀")
    st.markdown(
        """
        ### 🌟 About This App  
        ClimateGuard is an **AI-powered disaster prediction system** that helps analyze risks from:  
        - 🌊 **Floods** (using terrain + rainfall data)  
        - 🔥 **Wildfires** (using fire incident data from California)  

        ### ✅ Features  
        - Real-time **risk prediction** based on user inputs.  
        - 📊 **Interactive charts** to explore disaster data.  
        - 🔎 **Feature importance** to understand which factors matter most.  

        ### 🎯 Goal  
        To showcase how **AI & Machine Learning** can be applied to **climate disasters** and help with early risk assessment.  

        👉 Use the **sidebar** to select *Flood Prediction* or *Wildfire Prediction* and start exploring.
        """
    )

# -----------------------
# Flood Prediction
# -----------------------
elif option == "Flood Prediction 🌊":
    st.header("Flood Prediction AI")

    df = load_flood_data()
    model, X, y, X_test, y_test = train_flood_model(df)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    st.write(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

    # Flood Risk Distribution
    st.subheader("📊 Flood Susceptibility Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette="Blues", order=y.value_counts().index, ax=ax)
    ax.set_title("Flood Susceptibility Levels")
    ax.set_xlabel("Flood Risk Level")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("📊 Feature Importance in Flood Prediction")
    flood_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=flood_importances, palette="Blues", ax=ax2)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

    # User input
    st.subheader("Enter Values to Predict Flood Risk")
    slope = st.number_input("Slope", 0.0, 100.0, 40.0)
    curvature = st.number_input("Curvature", -5000000000.0, 5000000000.0, -1296000000.0)
    aspect = st.number_input("Aspect", 0.0, 360.0, 180.0)
    twi = st.number_input("TWI", -10.0, 20.0, 5.0)
    fa = st.number_input("FA", 0.0, 500.0, 100.0)
    drainage = st.number_input("Drainage", 0.0, 500.0, 250.0)
    rainfall = st.number_input("Rainfall", 0.0, 500.0, 80.0)

    if st.button("Predict Flood Risk"):
        sample = [[slope, curvature, aspect, twi, fa, drainage, rainfall]]
        prediction = model.predict(sample)
        st.success(f"🌊 Predicted Flood Risk: **{prediction[0]}**")

# -----------------------
# Wildfire Prediction
# -----------------------
elif option == "Wildfire Prediction 🔥":
    st.header("Wildfire Prediction AI")

    df = load_fire_data()
    model, X, y, X_test, y_test = train_fire_model(df)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    st.write(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

    # Wildfire Status Distribution
    st.subheader("📊 Wildfire Status Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x=y, palette="Oranges", order=y.value_counts().index, ax=ax3)
    ax3.set_title("Wildfire Status Levels")
    ax3.set_xlabel("Wildfire Status")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

    # Feature Importance
    st.subheader("📊 Feature Importance in Wildfire Prediction")
    fire_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig4, ax4 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fire_importances, palette="Oranges", ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

    # User input
    st.subheader("Enter Values to Predict Wildfire Status")
    acres = st.number_input("Acres Burned", 0.0, 1000000.0, 50000.0)
    latitude = st.number_input("Latitude", 32.0, 42.0, 37.5)
    longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)
    fatalities = st.number_input("Fatalities", 0, 100, 2)
    injuries = st.number_input("Injuries", 0, 500, 10)

    if st.button("Predict Wildfire Status"):
        sample = [[acres, latitude, longitude, fatalities, injuries]]
        prediction = model.predict(sample)
        st.success(f"🔥 Predicted Wildfire Status: **{prediction[0]}**")

# -----------------------
# Footer
# -----------------------
st.markdown(
    """
    ---
    💡 Built with ❤️ using **Python, Streamlit, and ML**  
    © 2025 Monica Yuol Manyok | ClimateGuard Project 🌍
    """,
    unsafe_allow_html=True
)
