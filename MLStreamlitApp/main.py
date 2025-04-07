import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, mean_squared_error, r2_score
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

# Streamlit Interface

# App title
st.set_page_config(page_title="DataFlex", layout="wide")
st.title("DataFlex")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload Data & Preprocessing", "Model Selection", "Results & Visualization"])

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
            params["max_depth"] = st.slider("Max Depth", 1, 10, 1)
            params["min_samples_split"] = st.slider("Min Samples per Split", None, 10, 2)
            params["min_samples_leaf"] = st.slider("Min Samples per Leaf", None, 5, 1)
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

# Tab 3: Evaluation & Comparison
with tab3:
    st.header("Model Evaluation and Comparison")

    if "results" in st.session_state:
        y_test = st.session_state["results"]["y_test"]
        y_pred = st.session_state["results"]["y_pred"]
        y_probs = st.session_state["results"]["y_probs"]
        model_name = st.session_state["results"]["model_name"]

        st.subheader(f"Performance of {model_name}")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2f}")
        pres = precision_score(y_test, y_pred, average='macro')
        st.metric("Precision", f"{pres:.2f}")
        rec = recall_score(y_test, y_pred, average='macro')
        st.metric("Recall", f"{rec:.2f}")
        fbeta = fbeta_score(y_test, y_pred, beta=1, average='macro')
        st.metric("F Score", f"{fbeta:.2f}")
        roc_auc = roc_auc_score(y_test, y_probs)
        st.metric("ROC AUC Score", f"{roc_auc:.2f}")

        plot_confusion_matrix(y_test, y_pred, model_name)

        with st.expander("Compare with another model"):
            model_name_2 = st.selectbox("Select comparison model", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"], key="compare_model")

            params_2 = {}
            st.markdown("### Set comparison model hyperparameters:")
            if model_name_2 == "Logistic Regression":
                params_2["C"] = 1.0
                params_2["max_iter"] = 300
            elif model_name_2 == "Decision Tree":
                params_2["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="max_depth_2")
            elif model_name_2 == "K-Nearest Neighbors":
                params_2["n_neighbors"] = st.slider("Number of Neighbors (k)", 1, 15, 5, key="n_neighbors_2")

            if st.button("Train and Compare"):
                model2 = get_model(model_name_2, params_2)
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model2.fit(X_train, y_train)
                y_pred_2 = model2.predict(X_test)
                y_probs_2 = model2.predict_proba(X_test)[:, 1]

                acc2 = accuracy_score(y_test, y_pred_2)
                pres2 = precision_score(y_test, y_pred_2)
                rec2 = recall_score(y_test, y_pred_2)
                f2 = recall_score (y_test, y_pred_2)
                roc_auc2 = roc_auc_score(y_test, y_probs_2)
                st.metric(f"{model_name_2} Accuracy", f"{acc2:.2f}")
                st.metric(f"{model_name_2} Precision", f"{pres2:.2f}")
                st.metric(f"{model_name_2} Recall", f"{rec2:.2f}")
                st.metric(f"{model_name_2} F-score", f"{f2:.2f}")
                st.metric("ROC AUC Score", f"{roc_auc2:.2f}")
                
                plot_confusion_matrix(y_test, y_pred_2, model_name_2)

    else:
        st.info("Train a model first in Tab 2 to view evaluations.")