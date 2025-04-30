import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Helper Functions
def load_data(file):
    df = pd.read_csv(file)
    return df

# Demo Data: Breast Cancer Wisconsin dataset
breast_cancer = load_breast_cancer()
breast_cancer_X = breast_cancer.data  # Feature matrix
breast_cancer_y = breast_cancer.target  # Target variable (diagnosis)
breast_cancer_feature_names = breast_cancer.feature_names
breast_cancer_target_names = breast_cancer.target_names

# Streamlit Interface

if 'welcome_screen' not in st.session_state:
    st.session_state['welcome_screen'] = True

if st.session_state['welcome_screen']:
    st.title("Welcome to DataQuest!")
    st.write("DataQuest allows you to upload datasets, group data with unsupervised machine learning models, and visualize the results.")
    st.write("Follow these steps to get started:")
    st.write("1. Upload a dataset or select a demo dataset.")
    st.write("2. Choose features and the target variable.")
    st.write("3. Train a model, tune hyperparameters, and view evaluation metrics.")
    st.write("4. Visualize model performance.")
    st.button("Get Started", on_click=lambda: st.session_state.update({'welcome_screen': False}))
else:
    st.set_page_config(page_title="DataQuest", layout="wide")
    st.title("DataQuest")

    # Sidebar for controls
    st.sidebar.header("Dataset Options")

    # Dataset selection
    data_source = st.sidebar.selectbox("Choose Dataset", ["Upload CSV", "Breast Cancer"])
    