import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Regression ML App",
    layout="wide"
)

st.title("📈 Regression Machine Learning App")
st.write(
    "Train, compare, and visualize different regression models "
    "using a real dataset."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("car_data.csv")
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    test_size = st.slider(
        "Test Set Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05
    )

    target = st.selectbox(
        "Select Target Column (Numeric)",
        df.columns
    )

# --------------------------------------------------
# DATA VALIDATION & CLEANING
# --------------------------------------------------
# Drop missing values
df = df.dropna()

# Check target type
if not pd.api.types.is_numeric_dtype(df[target]):
    st.error("❌ Target column must be NUMERIC for regression.")
    st.stop()

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Dataset Shape:** {df.shape}")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.write("**Statistical Summary (Numeric Columns)**")
    st.dataframe(df.describe(include="number"), use_container_width=True)

# --------------------------------------------------
# FEATURE / TARGET SPLIT
# --------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# Encode categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42
)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
st.subheader("🤖 Model Selection")

model_choice = st.radio(
    "Choose Regression Model",
    [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest Regression"
    ],
    horizontal=True
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
if model_choice == "Linear Regression":
    model = LinearRegression()

elif model_choice == "Ridge Regression":
    alpha = st.slider(
        "Alpha (Regularization Strength)",
        min_value=0.01,
        max_value=100.0,
        value=1.0
    )
    model = Ridge(alpha=alpha)

else:
    col1, col2, col3 = st.columns(3)

    with col1:
        n_estimators = st.slider(
            "Number of Trees",
            50, 500, 100
        )

    with col2:
        max_depth = st.selectbox(
            "Max Depth",
            [None, 10, 20, 30, 40]
        )

    with col3:
        min_samples_split = st.slider(
            "Min Samples Split",
            2, 20, 2
        )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
if st.button("🚀 Train Model", type="primary"):

    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    st.success("✅ Model trained successfully!")

    # --------------------------------------------------
    # METRICS DISPLAY
    # --------------------------------------------------
    st.subheader("📈 Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train R²", f"{r2_train:.4f}")
    m2.metric("Test R²", f"{r2_test:.4f}")
    m3.metric("RMSE", f"{rmse:.4f}")
    m4.metric("MAE", f"{mae:.4f}")

    # --------------------------------------------------
    # SAMPLE PREDICTIONS
    # --------------------------------------------------
    st.subheader("🎯 Sample Predictions")

    result_df = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": y_test_pred[:10],
        "Absolute Error": np.abs(
            y_test.values[:10] - y_test_pred[:10]
        )
    })

    st.dataframe(result_df, use_container_width=True)

    # --------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------
    st.subheader("📊 Visualizations")

    v1, v2 = st.columns(2)

    # Actual vs Predicted
    with v1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.scatter(y_test, y_test_pred, alpha=0.5)
        ax1.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--"
        )
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Actual vs Predicted")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # Residual Plot
    with v2:
        residuals = y_test.values - y_test_pred
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(y_test_pred, residuals, alpha=0.5)
        ax2.axhline(0, color="r", linestyle="--")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
