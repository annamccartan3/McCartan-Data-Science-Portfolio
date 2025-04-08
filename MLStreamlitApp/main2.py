import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Helper Functions
def get_model(name, params):
    if name == "Logistic Regression":
        model = LogisticRegression(C=params["C"], max_iter=params["max_iter"], multi_class='ovr')
    elif name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"])
    elif name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    return model

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix")
    st.pyplot(fig)

def plot_roc_curve(y_test, y_probs):
    plt.figure()

    # Binary classification
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    sns.lineplot(x=fpr, y=tpr, label=f"AUC = {roc_auc:.2f}")

    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

st.set_page_config(page_title="DataFlex", layout="wide")
st.title("DataFlex")

# Sidebar for controls
st.sidebar.header("Dataset Options")

# Dataset selection
data_source = st.sidebar.selectbox("Choose Dataset", ["Upload your own", "Titanic"])

if data_source == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.sidebar.subheader("Handle Missing Values")
        missing_option = st.sidebar.radio("Choose method:", ["None", "Drop Rows", "Impute (Mean/Mode)"])

        if missing_option == "Drop Rows":
            df = df.dropna()
        elif missing_option == "Impute (Mean/Mode)":
            imputer = SimpleImputer(strategy='mean')
            df_numeric = df.select_dtypes(include=['float64', 'int64'])
            df[df_numeric.columns] = imputer.fit_transform(df_numeric)
            for col in df.select_dtypes(include='object').columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
else:
    df = sns.load_dataset('titanic')
    df = df.dropna(subset=['age'], inplace=True)
    df = pd.get_dummies(demo_data, columns=['sex'], drop_first=True) # Use drop_first = True to avoid "dummy trap"
    df_kept = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male', 'survived']
    df = df[df_kept]

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload Data & Preprocessing", "Model Selection", "Visualization"])

with tab1:
    st.header("Choose a Dataset")
    if 'df' in locals():
        st.subheader("Dataset Preview")
        st.dataframe(df)

        st.sidebar.subheader("Feature Scaling")
        scaling_option = st.sidebar.radio("Choose Scaling Method:", ["None", "StandardScaler"])

        st.write("Shape of Dataset:", df.shape)
        st.write("Missing Values per Column:")
        st.write(df.isnull().sum())

    else:
        st.warning("Please upload a dataset or select a demo dataset.")

    # Choose target/feature variables
    df = st.session_state.df
    with st.expander("Select target variable"):
        target = st.selectbox("Choose target column", df.columns, index=(len(df.columns)-1))
    with st.expander("Select feature variables"):
        features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

    # Set training parameters
    st.header("Set Training Parameters")

    if 'df' in locals():
        test_size = st.slider("Test set size (fraction)", 0.1, 0.5, 0.2, step=0.05)
        random_state = st.number_input("Random state (seed)", value=42, step=1)

        # Store parameters in session_state
        st.session_state['test_size'] = test_size
        st.session_state['random_state'] = random_state

        st.success(f"Parameters saved! Test size: {test_size}, Random state: {random_state}")
    else:
        st.warning("Please select a dataset first.")

with tab2:
    st.header("Tab 2")
    # Option to scale data
    st.markdown("### Feature Scaling")
    scaling_option = st.selectbox("Choose scaling method:", ["None", "StandardScaler"])

    if st.button("Train Model"):
        X = df[features]
        y = df[target]

        # Use parameters from Tab 2
        test_size = st.session_state.get('test_size', 0.2)
        random_state = st.session_state.get('random_state', 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Scale data if selected
        if scaling_option == "StandardScaler":
            scaler = StandardScaler()
            numeric_cols = X_train.select_dtypes(include='number').columns

            # Fit only on training data
            X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

            # Transform test data
            X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

        model = get_model(model_name, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if len(np.unique(y)) == 2:
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            st.warning("ROC Curve only supported for binary classification.")
            y_probs = None

        st.success(f"{model_name} trained successfully!")
        st.session_state["results"] = {
            "model_name": model_name,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_probs": y_probs
        }
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader("Quick Performance Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
        with col2:
            st.metric("Precision", f"{precision:.2f}")
        col3, col4 = st.columns(2)
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")

with tab3:
    st.header("Tab 3")