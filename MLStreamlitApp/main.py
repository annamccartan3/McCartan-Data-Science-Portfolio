import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Demo Data
## Demo Data 1: Titanic
demo_data1 = sns.load_dataset('titanic')
# Handling missing values
demo_data1.dropna(subset=['age'], inplace=True)
# Encoding categorical variables
demo_data1 = pd.get_dummies(demo_data1, columns=['sex'], drop_first=True) # Use drop_first = True to avoid "dummy trap"
demo_data1_kept = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male', 'survived']
demo_data1 = demo_data1[demo_data1_kept]

## Demo Data 2: Iris
demo_data2 = sns.load_dataset('iris')

# Helper Functions

def load_data(file):
    df = pd.read_csv(file)
    return df

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
    ax.set_title(f"Confusion Matrix: {model_name}")
    st.pyplot(fig)

def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC Curve for binary or multiclass classification.
    """
    plt.figure(figsize=(8, 6))

    # Check if model supports probability prediction
    if not hasattr(model, "predict_proba"):
        st.warning("This model does not support probability predictions needed for ROC curve.")
        return

    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    # Binary classification
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        sns.lineplot(x=fpr, y=tpr, label=f"AUC = {roc_auc:.2f}")

    # Multiclass classification
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            sns.lineplot(x=fpr, y=tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")

    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Streamlit Interface

# App title
st.set_page_config(page_title="DataFlex", layout="wide")
st.title("DataFlex")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload Data & Preprocessing", "Model Selection", "Visualization"])

# Tab 1: Upload & Preprocess Data
with tab1:
    st.header("Choose a Dataset")
    dataset_option = st.radio("Select dataset source", ["Demo Dataset", "Upload CSV"])

    # custom dataset upload
    if dataset_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            st.success("Dataset loaded successfully!")
            st.dataframe(df.head())

            st.subheader("Data Preprocessing")

            st.markdown("### Handle Missing Values")
            missing_option = st.selectbox("Choose how to handle missing values:", ["Drop rows with missing values", "Fill with mean", "Fill with median"])

            if missing_option == "Drop rows with missing values":
                df = df.dropna()
            elif missing_option == "Fill with mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_option == "Fill with median":
                df = df.fillna(df.median(numeric_only=True))

            st.subheader("Processed Data Preview")
            st.write(df.head())

            st.session_state.df = df
    
    # or, choose a demo dataset
    else:
        demo_choice = st.selectbox("Choose a demo dataset", ["titanic", "iris"])
        if demo_choice == "titanic":
            df = demo_data1
        else:
            df = demo_data2
        st.success(f"{demo_choice.capitalize()} dataset loaded successfully!")
        st.dataframe(df.head())
   
    # option to scale data
    st.markdown("### Feature Scaling")
    scaling_option = st.selectbox("Choose scaling method:", ["None", "MinMaxScaler", "StandardScaler"])
    if scaling_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        df[df.select_dtypes(include='number').columns] = scaler.fit_transform(df.select_dtypes(include='number'))
    elif scaling_option == "StandardScaler":
        scaler = StandardScaler()
        df[df.select_dtypes(include='number').columns] = scaler.fit_transform(df.select_dtypes(include='number'))

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


# Tab 2: Model Selection & Training
with tab2:
    st.header("Choose and Train Your Model")

    # choose a target variable
    if 'df' in locals():
        with st.expander("Select target variable"):
            target = st.selectbox("Choose target column", df.columns, index=(len(df.columns)-1))
        with st.expander("Select feature variables"):
            features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])
    else:
        st.warning("Please choose a dataset to continue.")

    if 'df' in locals():
        model_name = st.selectbox("Select model", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])
        params = {}
        if model_name == "Logistic Regression":
            params["C"] = 1.0
            params["max_iter"] = 300
        if model_name == "Decision Tree":
            st.subheader("Set hyperparameters")
            params["max_depth"] = st.slider("Max Depth", 1, 10, 2)
            params["min_samples_split"] = st.slider("Min Samples per Split", 1, 10, 2)
            params["min_samples_leaf"] = st.slider("Min Samples per Leaf", 1, 10, 2)
        elif model_name == "K-Nearest Neighbors":
            st.subheader("Set hyperparameters")
            params["n_neighbors"] = st.slider("Number of Neighbors (k)", 1, 15, 5)

        if st.button("Train Model"):
            X = df[features]
            y = df[target]

            # Use parameters from Tab 2
            test_size = st.session_state.get('test_size', 0.2)
            random_state = st.session_state.get('random_state', 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

            model = get_model(model_name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]

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

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{precision:.2f}")
            col3.metric("Recall", f"{recall:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")

# Tab 3: Visualization
with tab3:
    st.header("Model Visualization")

    if "results" in st.session_state:
        y_test = st.session_state["results"]["y_test"]
        y_pred = st.session_state["results"]["y_pred"]
        y_probs = st.session_state["results"]["y_probs"]
        model_name = st.session_state["results"]["model_name"]

        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(model, X_test, y_test)

    else:
        st.info("Train a model first in Tab 2 to view evaluations.")