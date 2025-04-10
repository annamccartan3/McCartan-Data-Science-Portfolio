import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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
    ax.set_title(f"Confusion Matrix")
    st.pyplot(fig)

def plot_roc_curve(y_test, y_probs):
    if y_probs is not None:
        plt.figure()

        # Plot ROC Curve
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
    else:
        st.warning("ROC Curve only supported for binary classification.")

# Demo Data: Titanic
demo = sns.load_dataset('titanic') # load dataset
demo.dropna(subset=['age'], inplace=True) # handle missing values
demo = pd.get_dummies(demo, columns=['sex'], drop_first=True) # encode categorical variable
demo_kept = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male', 'survived'] # select specific columns
demo = demo[demo_kept]

# Demo Data: Iris
demo_2 = sns.load_dataset('iris') # load dataset

# Streamlit Interface

if 'welcome_screen' not in st.session_state:
    st.session_state['welcome_screen'] = True

if st.session_state['welcome_screen']:
    st.title("Welcome to DataFlex!")
    st.write("DataFlex allows you to upload datasets, train machine learning models, and visualize the results.")
    st.write("Follow these steps to get started:")
    st.write("1. Upload a dataset or select a demo dataset.")
    st.write("2. Choose features and the target variable.")
    st.write("3. Train a model, tune hyperparameters, and evaluate performance.")
    st.write("4. Visualize model performance (confusion matrix, ROC curve).")
    st.button("Get Started", on_click=lambda: st.session_state.update({'welcome_screen': False}))
else:
    st.set_page_config(page_title="DataFlex", layout="wide")
    st.title("DataFlex")

    # Sidebar for controls
    st.sidebar.header("Dataset Options")

    # Dataset selection
    data_source = st.sidebar.selectbox("Choose Dataset", ["Upload CSV", "Titanic", "Iris"])

    # Custom dataset upload
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], help="Upload your dataset here. Make sure it’s in CSV format. The dataset should contain a target column for model training.")
        if uploaded_file:
            df = load_data(uploaded_file)
            st.success("Dataset loaded successfully!")

            # Handling missing values
            st.sidebar.subheader("Handle Missing Values")
            missing_option = st.sidebar.radio("Choose method:", ["None", "Drop Rows", "Impute Mean"])

            if missing_option == "Drop Rows":
                df = df.dropna()
            elif missing_option == "Impute Mean":
                imputer = SimpleImputer(strategy='mean')
                df_numeric = df.select_dtypes(include=['float64', 'int64'])
                df[df_numeric.columns] = imputer.fit_transform(df_numeric)
                for col in df.select_dtypes(include='object').columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)

            st.session_state.df = df

    # Or, choose a demo dataset
    else:
        if data_source == "Titanic":
            df = demo
        else:
            df = demo_2
        st.success(f"{data_source} dataset loaded successfully!")
        st.session_state.df = df

    # Splitting parameters
    if 'df' in locals():
        st.sidebar.subheader("Training Parameters")
        test_size = st.sidebar.slider("Test set size (fraction)", 0.1, 0.5, 0.2, step=0.05)
        random_state = st.sidebar.number_input("Random state (seed)", value=42, step=1)

        # Store parameters in session_state
        st.session_state['test_size'] = test_size
        st.session_state['random_state'] = random_state
        st.sidebar.success(f"Parameters saved! Test size: {test_size}, Random state: {random_state}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Refine Data", "Train Model", "Visualization"])

    # Tab 1: Refine Data
    with tab1:
        st.header("Refine Data")
        if 'df' in st.session_state:
            df = st.session_state.df

            # Choose target/feature variables
            st.subheader("Choose Variables")
            with st.expander("Select target variable"):
                target = st.selectbox("Choose target column", df.columns, index=(len(df.columns)-1))
            with st.expander("Select feature variables"):
                features = st.multiselect("Select feature columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

            # Display Dataframe
            kept_cols = features + [target]
            df = df[kept_cols]
            st.dataframe(df.head())

            # Option to scale data
            st.subheader("Scale Data")
            scaling_option = st.selectbox("Select scaling method:", ["None", "StandardScaler"])

        else:
            st.info("Please select a dataset first.")

    # Tab 2: Train Model
    with tab2:
        st.header("Train Model")
        if 'df' in st.session_state:
            df = st.session_state.df
            model_name = st.selectbox("Select model", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])
            params = {}
            if model_name == "Logistic Regression":
                params["C"] = 1.0
                params["max_iter"] = 300
            if model_name == "Decision Tree":
                st.subheader("Set Hyperparameters")
                params["max_depth"] = st.slider("Max Depth", 1, 10, 2, help="Max Depth: Controls the maximum depth of the tree. Deeper trees tend to overfit")
                params["min_samples_split"] = st.slider("Min Samples per Split", 1, 10, 2, help="Min Samples per Split: The minimum number of samples required to split an internal node. Higher values help limit overfitting.")
                params["min_samples_leaf"] = st.slider("Min Samples per Leaf", 1, 10, 2, help="Min Samples per Leaf: The minimum number of samples required to be at a leaf node. Higher values help limit overfitting.")
            elif model_name == "K-Nearest Neighbors":
                st.subheader("Set Hyperparameters")
                params["n_neighbors"] = st.slider("Number of Neighbors (k)", 1, 21, 5, help="Number of Neighbors (k): The number of neighbors to consider when making a prediction. A smaller K can lead to a model that's sensitive to noise, while a larger K can smooth out the decision boundary.")

            if st.button("Train Model"):
                X = df[features]
                y = df[target]

                # Use parameters from sidebar
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

                # Train model
                model = get_model(model_name, params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if len(np.unique(y)) == 2:
                    y_probs = model.predict_proba(X_test)[:, 1]
                else:
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
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                st.subheader("Performance Summary")
                with st.expander("What do these metrics mean?"):
                    st.markdown("""
                    - **Accuracy**: Proportion of total predictions that were correct.
                    - **Precision**: Of all predicted positives, how many were actually positive.
                    - **Recall**: Of all actual positives, how many were correctly predicted.
                    - **F1 Score**: Measures the balance between Precision & Recall (closer to 1 = better balance)
                    """)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2f}")
                with col2:
                    st.metric("Precision", f"{precision:.2f}")
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Recall", f"{recall:.2f}")
                with col4:
                    st.metric("F1 Score", f"{f1:.2f}")
                
                # If Logistic Regression was chosen, plot Feature Importance
                if model_name == "Logistic Regression":
                    st.subheader("Feature Importance")

                    coeff = pd.Series(model.coef_[0], index=X.columns)
                    coeff = coeff.sort_values()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=coeff.values, y=coeff.index, palette='coolwarm', ax=ax)
                    ax.set_title('Feature Importance (Coefficients)')
                    ax.set_ylabel('')
                    st.pyplot(fig)

                # If KNN was chosen, plot Accuracy vs k
                if model_name == "K-Nearest Neighbors":
                    st.subheader("KNN Visualization")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Define a range of k values to explore for all odd numbers
                        k_values = list(range(1, 21, 2))
                        accuracy_scores = []
                        # Loop through different values of k, train a KNN model on data, record accuracy
                        for k in k_values:
                            knn = KNeighborsClassifier(n_neighbors=k)
                            knn.fit(X_train, y_train)
                            y_pred = knn.predict(X_test)
                            accuracy_scores.append(accuracy_score(y_test, y_pred))
                        # Plot accuracy vs. number of neighbors (k)
                        fig, ax = plt.subplots()
                        ax.plot(k_values, accuracy_scores, marker='o')
                        ax.set_xlabel('Number of Neighbors: k')
                        ax.set_ylabel('Accuracy')
                        ax.set_title('Accuracy')
                        st.pyplot(fig)
                    
                    with col2:
                        # Define a range of k values to explore for all odd numbers
                        k_values = list(range(1, 21, 2))
                        f1_scores = []
                        # Loop through different values of k, train a KNN model on data, record F1 score
                        for k in k_values:
                            knn = KNeighborsClassifier(n_neighbors=k)
                            knn.fit(X_train, y_train)
                            y_pred = knn.predict(X_test)
                            f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
                        # Plot F1 vs. number of neighbors (k)
                        fig, ax = plt.subplots()
                        ax.plot(k_values, f1_scores, marker='o')
                        ax.set_xlabel('Number of Neighbors: k')
                        ax.set_ylabel('F1 Score')
                        ax.set_title('F1 Score')
                        st.pyplot(fig)

                # If Decision was chosen, plot Decision Tree Visual
                if model_name == "Decision Tree":
                        st.subheader("Decision Tree Visualization")
                        dot_data = export_graphviz(model,
                                                feature_names=X.columns,
                                                class_names=[str(cls) for cls in model.classes_],
                                                filled=True)

                        st.graphviz_chart(dot_data)


        else:
            st.info("To train a model, please select a dataset first.")
        
    # Tab 3: Visualization
    with tab3:
        st.header("Visualization")
        if "results" in st.session_state:
            y_test = st.session_state["results"]["y_test"]
            y_pred = st.session_state["results"]["y_pred"]
            y_probs = st.session_state["results"]["y_probs"]
            model_name = st.session_state["results"]["model_name"]

            st.subheader(model_name)

            # Set up two columns for side-by-side visualization
            col1, col2 = st.columns(2)
            with col1:
                plot_confusion_matrix(y_test, y_pred, model_name)      
                with st.expander("What does this metric mean?"):
                    st.markdown("""
                    - **Confusion Matrix**: The Confusion Matrix shows how many of your model's predictions were correct (True Positives/Negatives, along the diagonal) and how many were incorrect (False Positives/Negatives).
                    """)
            with col2:
                plot_roc_curve(y_test, y_probs)
                with st.expander("What does this metric mean?"):
                    st.markdown("""
                    - **ROC Curve**: The ROC Curve shows how well your model can distinguish between positive and negative cases. The higher the Area Under the Curve (AUC), the better your model is at making this distinction.
                    """)  

        else:
            st.info("To view visualization, train a model in Tab 2 first.")