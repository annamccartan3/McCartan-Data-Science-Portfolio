import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions ---

def load_data(file):
    df = pd.read_csv(file)
    return df

def get_model(name, params):
    if name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
    elif name == "Support Vector Machine":
        model = SVC(C=params["C"], kernel=params["kernel"], probability=True)
    elif name == "Logistic Regression":
        model = LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    return model

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    st.pyplot(fig)

# --- Streamlit Interface ---

st.set_page_config(page_title="ML Explorer", layout="wide")
st.title("🧠 Interactive Machine Learning Explorer")

tab1, tab2, tab3 = st.tabs(["1. Dataset Upload", "2. Model Selection & Training", "3. Evaluation & Comparison"])

# --- Tab 1: Dataset Upload ---
with tab1:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.dataframe(df.head())

        with st.expander("Select target variable"):
            target = st.selectbox("Choose target column", df.columns)
            features = [col for col in df.columns if col != target]
    else:
        st.warning("Please upload a dataset to continue.")

# --- Tab 2: Model Selection ---
with tab2:
    st.header("Choose and Train Your Model")

    if uploaded_file is not None:
        model_name = st.selectbox("Select model", ["Random Forest", "Support Vector Machine", "Logistic Regression"])

        st.subheader("Set hyperparameters")
        params = {}
        if model_name == "Random Forest":
            params["n_estimators"] = st.slider("Number of trees", 10, 200, 100)
            params["max_depth"] = st.slider("Max depth", 1, 20, 5)
        elif model_name == "Support Vector Machine":
            params["C"] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
            params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        elif model_name == "Logistic Regression":
            params["C"] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
            params["max_iter"] = st.slider("Max Iterations", 100, 1000, 300)

        if st.button("Train Model"):
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = get_model(model_name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"{model_name} trained successfully!")
            st.session_state["results"] = {
                "model_name": model_name,
                "y_test": y_test,
                "y_pred": y_pred
            }

# --- Tab 3: Evaluation & Comparison ---
with tab3:
    st.header("Model Evaluation and Comparison")

    if "results" in st.session_state:
        y_test = st.session_state["results"]["y_test"]
        y_pred = st.session_state["results"]["y_pred"]
        model_name = st.session_state["results"]["model_name"]

        st.subheader(f"Performance of {model_name}")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, model_name)

        with st.expander("Compare with another model"):
            model_name_2 = st.selectbox("Select comparison model", ["Support Vector Machine", "Random Forest", "Logistic Regression"], key="compare_model")

            if st.button("Train and Compare"):
                # Simple reuse of the previous workflow
                params_2 = {"n_estimators": 100, "max_depth": 5, "C": 1.0, "kernel": "rbf", "max_iter": 300}
                model2 = get_model(model_name_2, params_2)
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model2.fit(X_train, y_train)
                y_pred_2 = model2.predict(X_test)

                acc2 = accuracy_score(y_test, y_pred_2)
                st.metric(f"{model_name_2} Accuracy", f"{acc2:.2f}")
                plot_confusion_matrix(y_test, y_pred_2, model_name_2)

                st.text(f"{model_name_2} Classification Report")
                st.text(classification_report(y_test, y_pred_2))

    else:
        st.info("Train a model first in Tab 2 to view evaluations.")
