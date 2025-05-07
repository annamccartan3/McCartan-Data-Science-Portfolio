# DataQuest App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# add tooltips, metric summaries, and helpful user advice
# add thorough comments and organization

# Helper Functions

def get_color_palette(name="Set2", n=10):
    """Load Set2 Color Palette"""
    import seaborn as sns
    return sns.color_palette(name, n)

def load_data(file): 
    """Load CSV file into a DataFrame."""
    df = pd.read_csv(file)
    return df

def get_model(name, params):
    """
    Returns an initialized unsupervised model based on the model name and parameters.
    """
    if name == "Principal Component Analysis":
        model = PCA(n_components=params["n_components"], random_state=st.session_state['random_state'])
    elif name == "K-Means Clustering":
        model = KMeans(n_clusters=params["n_clusters"], random_state=st.session_state['random_state'])
    elif name == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params['linkage'])
    return model

def drop_and_encode_features(df, target_col):
    """
    Identifies columns with high missingness or non-numeric values, recommends dropping them or offers option to encode.
    """
    st.sidebar.subheader("Drop or Encode Incompatible Data")

    feature_cols = df.drop(columns=[target_col])  # Only features
    non_numeric_cols = feature_cols.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    high_missing_cols = feature_cols.columns[feature_cols.isnull().mean() > 0.5].tolist()

    recommended_drop = list(set(non_numeric_cols + high_missing_cols))

    encode_cols = []
    if non_numeric_cols:
        encode_cols = st.sidebar.multiselect(
            "Non-numeric columns to encode (One-Hot):",
            options=[col for col in non_numeric_cols],
            default=non_numeric_cols
        )

    drop_cols = st.sidebar.multiselect(
        "Columns recommended to drop:",
        options=feature_cols.columns.tolist(),
        default=[col for col in recommended_drop if col not in encode_cols]
    )

    if non_numeric_cols:
        st.sidebar.caption(f"Non-numeric columns: {', '.join(non_numeric_cols)}")
    if high_missing_cols:
        st.sidebar.caption(f"Columns with >50% missing: {', '.join(high_missing_cols)}")

    df_cleaned = df.drop(columns=drop_cols)
    if encode_cols:
        df_encoded = pd.get_dummies(df_cleaned, columns=encode_cols, drop_first=True)
        st.sidebar.success(f"Encoded {len(encode_cols)} column(s) via One-Hot Encoding.")
    else:
        df_encoded = df_cleaned

    return df_encoded, drop_cols, encode_cols

def fit_pca(X_scaled, y, selected_features=None, n_components=2):
    """
    This function fits a PCA model on the provided data, ensuring that the features and labels have matching shapes.
    It also handles feature selection if a subset of features is specified.

    Parameters:
    - X_scaled (array-like): The feature matrix (scaled or original).
    - y (array-like): The target labels or classes.
    - selected_features (list, optional): A list of selected feature indices to subset the features. Default is None.
    - n_components (int): The number of components to keep in PCA.

    Returns:
    - X_pca (array-like): The transformed data in PCA space.
    - pca (PCA object): The fitted PCA model.
    - y_subset (array-like): The corresponding target labels for the subset used in PCA.
    """

    # Check if selected features are provided
    if selected_features is not None:
        # Ensure selected features are valid indices
        X_subset = X_scaled[:, selected_features]  # Filter `X` based on selected features
        y_subset = y  # No filtering on `y`, just ensuring it matches X_subset
    else:
        X_subset = X_scaled  # Use full dataset if no feature selection
        y_subset = y  # Corresponding full target labels

    # Ensure that X and y have the same number of samples
    if len(X_subset) != len(y_subset):
        st.error(f"Feature set (X) and target labels (y) must have the same number of samples. X has {len(X_subset)}, but y has {len(y_subset)}.")
        return None, None, None

    # Apply PCA to the subset (or full data)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_subset)  # Fit PCA and transform the data

    # Save the PCA result and the corresponding target labels in session state
    st.session_state["X_pca"] = X_pca
    st.session_state["y_pca"] = y_subset

    return X_pca, pca, y_subset

def plot_pca_projection(X_pca, labels, label_names, title, palette):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for color, label, name in zip(palette, unique_labels, label_names):
        ax.scatter(
            X_pca[labels == label, 0],
            X_pca[labels == label, 1],
            color=color,
            label=name,
            edgecolor="black",
            alpha=0.7,
            s=60,
        )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    return fig

def plot_pca_projection_with_biplot(X_pca, y, target_names, feature_names, model, show_loadings=False, scaling_factor=50):
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    palette = sns.color_palette("Set2", num_classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    for color, class_val, label in zip(palette, unique_classes, target_names):
        ax.scatter(
            X_pca[y == class_val, 0], X_pca[y == class_val, 1],
            color=color, alpha=0.7, label=label,
            edgecolor='black', s=60
        )

    # Biplot arrows
    if show_loadings and hasattr(model, "components_") and model.components_.shape[0] >= 2:
        loadings = model.components_.T
        for i, feature in enumerate(feature_names):
            x_loading = scaling_factor * loadings[i, 0]
            y_loading = scaling_factor * loadings[i, 1]
            ax.arrow(0, 0, x_loading, y_loading, width=0.005, head_width=0.1,
                     alpha=0.7, length_includes_head=True)
            ax.text(x_loading * 1.15, y_loading * 1.15, feature,
                    fontsize=9, ha='center', va='center')

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("2D Projection of Dataset")
    ax.legend(loc="best")
    ax.grid(True)
    ax.axis('equal')

    return fig

def plot_pca_variance_explained(pca, color1="#66c2a5", color2="#fc8d62"):
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(explained) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(components, explained, alpha=0.8, label='Individual Variance', color=color1)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)')
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])

    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, marker='o', label='Cumulative Variance', color=color2)
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    fig.suptitle('PCA: Variance Explained', y=1.02)
    fig.tight_layout()
    return fig

def plot_hierarchical_clustering(X_scaled, cluster_labels, colors=None):
    """
    This function handles the plotting of the agglomerative hierarchical clustering result.
    
    Parameters:
    - X_scaled: The scaled feature matrix (used for PCA).
    - cluster_labels: The cluster labels assigned to each data point.
    - colors: Optional list of colors to use for the clusters.

    Returns:
    - None (shows plot via Streamlit).
    """
    # PCA Projection (fit & transform only once, if PCA is not already available)
    if "pca" not in st.session_state:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        st.session_state["pca"] = pca
        st.session_state["X_pca"] = X_pca
    else:
        X_pca = st.session_state["X_pca"]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_unique = np.unique(cluster_labels)

    # Define color palette if not provided
    if colors is None:
        colors = plt.cm.get_cmap('Set2', len(labels_unique))  # Default color palette
    
    for i, label in enumerate(labels_unique):
        idx = np.where(cluster_labels == label)
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=f"Cluster {label}",
            color=colors(i),
            edgecolor="black",
            alpha=0.7,
            s=60,
        )

    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title("Agglomerative Clustering on Data (via PCA)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True)
    st.pyplot(fig)

def show_welcome_screen():
    
    st.title("Welcome to DataQuest!")
    st.markdown(
        """
        DataQuest allows you to upload datasets, group data with unsupervised machine learning models, and visualize the results.
        
        **Steps to get started:**
        1. Upload a dataset or select a demo dataset.
        2. Choose features and the target variable.
        3. Train a model and tune hyperparameters.
        4. Visualize model performance.
        """
    )
    st.button("Get Started", on_click=lambda: st.session_state.update({'welcome_screen': False}))

st.set_page_config(page_title="DataQuest", layout="wide")

# ==== WELCOME SCREEN ====
if st.session_state.get('welcome_screen', True):
    show_welcome_screen()
    st.stop()

st.title("DataQuest")

# ==== SIDEBAR: DATA UPLOAD & SELECTION ====
st.sidebar.header("Dataset Options")

# Dataset selection
data_source = st.sidebar.selectbox("Choose Dataset", ["Upload CSV", "Breast Cancer", "Countries"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['df_raw'] = df.copy()
        st.success("Dataset loaded successfully!")

        st.sidebar.subheader("Select Target Variable")
        target_col = st.sidebar.selectbox("Select target column", df.columns, index=len(df.columns) - 1)
        st.session_state['target_col'] = target_col

        df_preprocessed, drop_cols, encoded_cols = drop_and_encode_features(df, target_col)

        st.session_state['df'] = df_preprocessed
        st.session_state['drop_cols'] = drop_cols
        st.session_state['encoded_cols'] = encoded_cols

        st.sidebar.subheader("Handle Missing Data")
        current_df = df_preprocessed.drop(columns=[target_col])
        if current_df.isnull().values.any():
            missing_option = st.sidebar.radio(
                "Missing values detected. How would you like to handle them?",
                ("Drop rows", "Impute mean"),
                key="missing_value_option"
            )
        else:
            missing_option = None
            st.sidebar.success("No missing values detected.")
        st.session_state['missing_option'] = missing_option

else:
    if data_source == "Breast Cancer":
        breast_cancer = load_breast_cancer(as_frame=True)
        df = breast_cancer.frame
        st.session_state['df_raw'] = df.copy()
        st.session_state['X'] = breast_cancer.data
        st.session_state['y'] = breast_cancer.target
        st.session_state['target_names'] = breast_cancer.target_names.tolist()
    if data_source == "Countries":
        file_path = 'data/Country-data.csv'
        df = pd.read_csv(file_path)
        st.session_state['df_raw'] = df.copy()
        st.session_state['X'] = df.drop(columns = "country")

    st.success(f"{data_source} dataset loaded successfully!")
    st.session_state['df'] = df
    st.session_state['drop_cols'] = None
    st.session_state['missing_option'] = None

    st.sidebar.subheader("Select Target Variable")
    # Dynamically set default target column index
    if data_source == "Breast Cancer" and "target" in df.columns:
        default_target_index = df.columns.get_loc("target")
    elif data_source == "Countries" and "country" in df.columns:
        default_target_index = df.columns.get_loc("country")
    else:
        default_target_index = len(df.columns) - 1  # fallback to last column
    
    # Selectbox with dynamic default
    target_col = st.sidebar.selectbox("Select target column", df.columns, index=default_target_index)
    st.session_state['target_col'] = target_col

# === Feature Selection & Training Parameters ===
if 'df' in st.session_state:
    df = st.session_state['df']
    target_col = st.session_state['target_col']

    st.sidebar.subheader("Select Feature Variables")
    feature_candidates = [col for col in df.columns if col != target_col]
    default_features = feature_candidates

    selected_features = st.sidebar.multiselect("Select feature columns", feature_candidates, default=default_features)
    st.session_state['selected_features'] = selected_features

    st.session_state['X'] = df[selected_features]
    st.session_state['y'] = df[target_col]
    st.session_state['kept_cols'] = selected_features + [target_col]

    st.sidebar.subheader("Set Training Parameters")
    random_state = st.sidebar.number_input("Random state (seed)", value=42, step=1)
    st.session_state['random_state'] = random_state
    st.sidebar.success(f"Random state set to: {random_state}")

# Store source name
st.session_state['data_source'] = data_source

# Display preview and class count
if 'df' in st.session_state:
    df_raw = st.session_state['df_raw']
    st.dataframe(df_raw.head())
    y_values = st.session_state['y']
    num_classes = len(np.unique(y_values))
    st.session_state['num_classes'] = num_classes
    st.write(f"Number of unique classes in target: {num_classes}")

# Tabs
tab1, tab2, tab3 = st.tabs(["Refine Data", "Train Model", "Visualization"])

# ==== TAB 1: DATA PREPROCESSING ====
with tab1:
    st.header("Refine Data")
    if 'df' in st.session_state:
        df = st.session_state['df']
        data_source = st.session_state['data_source']
        missing_option = st.session_state['missing_option']

        # Handle missing values
        if missing_option != None:
                if missing_option == "Drop rows":
                    df.dropna(inplace=True)
                    st.sidebar.success("Rows with missing values have been dropped.")
                elif missing_option == "Impute mean":
                    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.sidebar.success("Missing numeric values have been imputed with the column mean.")

        # Display Dataframe
        st.write("Preview of processed dataset:")
        kept_cols = st.session_state['kept_cols']
        df = df[kept_cols]
        st.dataframe(df.head())

        X = st.session_state['X']
        y = st.session_state['y']

        # Scale data
        st.subheader("Scale Data")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state['X_scaled'] = X_scaled # Store in session_state
        st.success(f"Data scaled automatically!")
        st.info("💡 Unsupervised methods work best when features are on the same scale, so that no single variable dominates the analysis.")
    else:
        st.info("Please select a dataset first.")

    # ==== TAB 2: MODEL TRAINING ====
    with tab2:
        st.header("Train Model")
        if 'df' in st.session_state:
            df = st.session_state['df']
            features = st.session_state.get('features', [])
            target = st.session_state.get('target', None)
            X_scaled = st.session_state['X_scaled']
            model_name = st.selectbox("Select model", ["Principal Component Analysis", "K-Means Clustering", "Hierarchical Clustering"])
            params = {}

            st.session_state["model_name"] = model_name
            
            # If PCA was chosen, display cumulative explained variance
            if model_name == "Principal Component Analysis":
                n_samples, n_features = X_scaled.shape
                min_components = min(20, n_features)
                pca_full = PCA(n_components=min_components).fit(X_scaled)
                explained_variance = pca_full.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)

                # Streamlit layout
                st.subheader("PCA Explained Variance")

                col1, col2 = st.columns(2)

                # Bar Chart
                with col1:
                    fig1, ax1 = plt.subplots()
                    ax1.bar(range(1, len(explained_variance)+1), explained_variance)
                    ax1.set_xlabel('Principal Component')
                    ax1.set_ylabel('Explained Variance Ratio')
                    ax1.set_title('Explained Variance by Component')
                    ax1.set_xticks(range(1, len(explained_variance)+1))
                    st.pyplot(fig1)

                # Cumulative Variance Plot
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--', color=get_color_palette()[1])
                    ax2.set_xlabel('Number of Components')
                    ax2.set_ylabel('Cumulative Explained Variance')
                    ax2.set_title('Cumulative Explained Variance')
                    ax2.set_xticks(range(1, len(cumulative_variance)+1))
                    ax2.grid(True)
                    st.pyplot(fig2)
                    # Explain the purpose of this plot
                    st.info("💡 Look for the 'elbow' in the cumulative curve to determine an optimal number of components.")
            
                params["n_components"] = st.slider("Number of Components", 1, min_components, 2, help="Number of Components: Choose the number of components to retain. Higher values capture more variance but may reduce interpretability.")
                
            # If KMeans was chosen, output the Elbow plot & Silhouette scores 
            elif model_name == "K-Means Clustering":
                ks = list(range(2, 11))
                wcss = []
                silhouette_scores = []

                for k in ks:
                    kmeans = KMeans(n_clusters=k, random_state=st.session_state["random_state"])
                    labels = kmeans.fit_predict(X_scaled)
                    wcss.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(X_scaled, labels))

                # Store results
                st.session_state["ks"] = ks
                st.session_state["wcss"] = wcss
                st.session_state["silhouette_scores"] = silhouette_scores

                st.subheader("K-Means Clustering Optimization")

                col1, col2 = st.columns(2)

                # Check that metrics are available
                if "wcss" in st.session_state and "silhouette_scores" in st.session_state and "ks" in st.session_state:
                    ks = st.session_state["ks"]
                    wcss = st.session_state["wcss"]
                    silhouette_scores = st.session_state["silhouette_scores"]

                    # Elbow Plot
                    with col1: 
                        fig1, ax1 = plt.subplots()
                        ax1.plot(ks, wcss, marker='o', color=get_color_palette()[3])
                        ax1.set_xlabel('Number of clusters (k)')
                        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
                        ax1.set_title('Elbow Method')
                        ax1.grid(True)
                        st.pyplot(fig1)
                        # Explain the purpose of this plot
                        st.info("💡 Ideal clustering minimizes the Within-Cluster Sum of Squares (WCSS). Look for the 'elbow' point of sharp decrease to determine an optimal k value.")

                    # Silhouette Plot
                    with col2:
                        fig2, ax2 = plt.subplots()
                        ax2.plot(ks, silhouette_scores, marker='o', color=get_color_palette()[4])
                        ax2.set_xlabel('Number of clusters (k)')
                        ax2.set_ylabel('Silhouette Score')
                        ax2.set_title('Silhouette Analysis')
                        ax2.grid(True)
                        st.pyplot(fig2)
                        # Explain the purpose of this plot
                        st.info("💡 The sillhouette score quantifies how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering.")
                
                    params["n_clusters"] = st.slider("Number of Clusters (k)", 1, 10, min(len(np.unique(y)), 10), help="Number of Clusters: [insert description here]")
                
                else:
                    st.warning("Please compute WCSS and Silhouette Scores before visualizing.")
                                
            # If Hierarchical was chosen, output the dendogram w/ truncation options
            elif model_name == "Hierarchical Clustering":

                params["linkage"] = st.selectbox("Choose Linkage Rule", ["ward", "single", "complete", "average"])

                ks = list(range(2, 11))
                silhouette_scores = []

                for k in ks:
                    agg = AgglomerativeClustering(n_clusters=k, linkage=params["linkage"])
                    labels = agg.fit_predict(X_scaled)
                    silhouette_scores.append(silhouette_score(X_scaled, labels))

                # Store results
                st.session_state["ks"] = ks
                st.session_state["silhouette_scores"] = silhouette_scores

                st.subheader("Hierarchical Clustering Optimization")

                # Check that metrics are available
                if "silhouette_scores" in st.session_state and "ks" in st.session_state:
                    ks = st.session_state["ks"]
                    silhouette_scores = st.session_state["silhouette_scores"]

                    # Silhouette Plot
                    fig2, ax2 = plt.subplots(figsize=(10,4))
                    ax2.plot(ks, silhouette_scores, marker='o', colors=get_color_palette()[4])
                    ax2.set_xlabel('Number of clusters (k)')
                    ax2.set_ylabel('Silhouette Score')
                    ax2.set_title("Silhouette Analysis")
                    ax2.grid(True)
                    st.pyplot(fig2)
                    # Explain the purpose of this plot
                    st.info("💡 The sillhouette score quantifies how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering.")

                # Slider for the number of clusters to display if truncation is enabled
                st.session_state['p_value'] = st.slider("Number of clusters to display in dendrogram (p)",
                                    min_value=2,
                                    max_value=max(st.session_state['num_classes'], 20),
                                    value=max(st.session_state['num_classes'], 20))
                truncate_mode = "lastp"

                # Safe label selection from dataset
                if "target" in df.columns and 'target_names' in st.session_state:
                    labels = [st.session_state['target_names'][val] for val in df["target"]]
                elif target_col in df.columns:
                    labels = df[target_col].tolist()
                else:
                    labels = df.index.astype(str).tolist()

                # Linkage and dendrogram plot
                Z = linkage(X_scaled, method=params["linkage"])
                fig1, ax1 = plt.subplots(figsize=(15,5))
                dendrogram(
                    Z,
                    labels=labels,  # Replace with your actual labels if necessary
                    truncate_mode=truncate_mode, 
                    p = st.session_state['p_value']
                )
                ax1.set_title("Dendrogram", fontsize=16)
                ax1.set_xlabel(target_col.capitalize(), fontsize=14)
                ax1.set_ylabel("Distance", fontsize=14)

                # Rotate x-tick labels if needed
                if ax1.get_xticks().size > 0:
                    plt.setp(ax1.get_xticklabels(), rotation=90)
                    plt.setp(ax1.get_yticklabels(), fontsize=12)

                st.pyplot(fig1)
                
                params["n_clusters"] = st.slider("Number of Clusters (k)", 1, 10, 3, help="Number of Clusters: [insert description here]")


            if st.button("Train Model"): # Train model

                model = get_model(model_name, params)
                st.session_state["model"] = model
                y = st.session_state['y']
                st.success(f"{model_name} trained successfully!")   
                
                # If PCA was chosen, display cumulative explained variance
                if model_name == "Principal Component Analysis":
                    X_pca = model.fit_transform(X_scaled)
                    st.session_state['X_pca'] = X_pca

                    explained_var = model.explained_variance_ratio_
                    cum_var = np.cumsum(explained_var)
                    total_cum_var = cum_var[-1]  # Last value = total cumulative variance
                    st.metric("Cumulative Explained Variance", f"{total_cum_var * 100:.2f}%")

                # If KMeans was chosen
                if model_name == "K-Means Clustering":
                    clusters = model.fit_predict(X_scaled)
                    st.session_state['clusters'] = clusters 

                    col1, col2 = st.columns(2)
                    with col1:
                        wcss = model.inertia_
                        st.metric("WCSS", f"{wcss:.2f}")
                    with col2:
                        sil_score = silhouette_score(X_scaled, clusters)
                        st.metric("Silhouette Score", f"{sil_score:.2f}")   

                # If Hierarchical was chosen
                if model_name == "Hierarchical Clustering":
                    df["Cluster"] = model.fit_predict(X_scaled)
                    st.session_state['df']['Cluster'] = df["Cluster"]
                    cluster_labels = df["Cluster"].tolist()
                    st.session_state['cluster_labels'] = cluster_labels   

                    sil_score = silhouette_score(X_scaled, df["Cluster"])
                    st.metric("Silhouette Score", f"{sil_score:.2f}")         

        else:
            st.info("To train a model, please select a dataset first.")

    # ==== TAB 3: VISUALIZATION ====  
    with tab3:
        st.header("Visualization")
        if 'model_name' not in st.session_state:
            st.info("To view visualization, train a model in Tab 2 first.")
        else:
            st.subheader(st.session_state.get("model_name"))

        # ==== PCA ====  
        if st.session_state.get("model_name") == "Principal Component Analysis":

            st.subheader("2D Projection")

            if "X_pca" in st.session_state and "y" in st.session_state:
                y = st.session_state["y"]
                num_classes = st.session_state["num_classes"]
                MAX_CLASSES = 10

                if num_classes > MAX_CLASSES:
                    st.warning(f"PCA plot is only supported for up to {MAX_CLASSES} classes. Your data has {num_classes}.")
                else:
                    target_names = st.session_state.get("target_names", [str(label) for label in np.unique(y)])
                    X_pca = st.session_state["X_pca"]
                    model = st.session_state.get("model")
                    feature_names = st.session_state.get("selected_features", [])

                    show_loadings = st.checkbox("Show PCA Feature Loadings (Biplot Arrows)")
                    scaling_factor = st.slider("Adjust arrow scaling", 1.0, 100.0, 50.0, step=1.0) if show_loadings else 50

                    fig = plot_pca_projection_with_biplot(
                        X_pca=X_pca,
                        y=y,
                        target_names=target_names,
                        feature_names=feature_names,
                        model=model,
                        show_loadings=show_loadings,
                        scaling_factor=scaling_factor
                    )
                    st.pyplot(fig)
            else:
                st.warning("Please train the PCA model in the 'Train PCA' tab first.")

            # Variance Explained Plot
            if "X_scaled" in st.session_state and "params" in st.session_state:
                X_scaled = st.session_state["X_scaled"]
                params = st.session_state["params"]
                pca = PCA(n_components=params["n_components"])
                pca.fit(X_scaled)
                fig = plot_pca_variance_explained(pca)
                st.subheader("Explained Variance")
                st.pyplot(fig)
                st.info("💡 Look for the 'elbow' in the cumulative curve to determine an optimal number of components.")
        
        # ==== K-Means Clustering ====  
        if st.session_state.get("model_name") == "K-Means Clustering":
            X_scaled = st.session_state.get("X_scaled")
            n_clusters = st.session_state.get("params", {}).get("n_clusters", 2)
            color_mode = st.selectbox("Color points by", ["K-Means Clusters", "Target Labels"])

            if color_mode == "K-Means Clusters":
                if "kmeans" not in st.session_state or \
                st.session_state.get("params", {}).get("n_clusters", None) != n_clusters:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=st.session_state['random_state'])
                    clusters = kmeans.fit_predict(X_scaled)
                    st.session_state["kmeans"] = kmeans
                    st.session_state["clusters"] = clusters
                    st.session_state["params"] = {"n_clusters": n_clusters}
                clusters = st.session_state["clusters"]
                labels = clusters
                label_names = [f"Cluster {i}" for i in np.unique(labels)]
            else:
                y = st.session_state.get("y")
                labels = y
                label_names = st.session_state.get("target_names", [str(i) for i in np.unique(y)])

            pca, X_pca = fit_pca(X_scaled)
            st.session_state["pca"] = pca
            palette = get_color_palette("tab10", len(np.unique(labels)))
            fig = plot_pca_projection(X_pca, labels, label_names, f"PCA Projection Colored by {color_mode}", palette)
            st.pyplot(fig)
            
        # ==== Hierarchical Clustering ====  
        if st.session_state.get("model_name") == "Hierarchical Clustering":
            if "cluster_labels" in st.session_state and "X_scaled" in st.session_state:
                X_scaled = st.session_state["X_scaled"]
                cluster_labels = st.session_state["cluster_labels"]

                # Plot the hierarchical clustering result
                plot_hierarchical_clustering(X_scaled, cluster_labels)
            else:
                st.warning("Cluster labels or scaled data not found. Please ensure clustering is performed first.")