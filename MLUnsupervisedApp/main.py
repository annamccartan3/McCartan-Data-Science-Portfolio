# DataQuest App

# ==== IMPORT NECESSARY PACKAGES ====
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ==== HELPER FUNCTIONS ====
def get_color_palette(name="Set2", n=10):
    """
    Returns list of RGB tuples representing a Seaborn color palette.
    """
    return sns.color_palette(name, n)

def show_welcome_screen():
    """
    Displays the welcome screen in the Streamlit app with instructions and a 'Get Started' button.
    """
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

    # Center the button below
    st.button("Get Started", on_click=lambda: st.session_state.update({'welcome_screen': False}))

def load_data(file): 
    """
    Loads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None
    return df

def handle_uploaded_data(uploaded_file):
    """
    Preprocesses custom CSV files for use with unsupervised ML models.
    """
    #Ensure Clean State
    for key in ['y', 'target_names', 'pca_results', 'visualization_data', 'cluster_labels', 'X_pca']:
        st.session_state.pop(key, None)

    df = load_data(uploaded_file)
    st.session_state['df_raw'] = df.copy()

    target_col = st.sidebar.selectbox("Select target column", df.columns, index=len(df.columns) - 1)
    st.session_state['target_col'] = target_col

    df_preprocessed, drop_cols, encoded_cols = drop_and_encode_features(df, target_col)

    st.session_state.update({
        'df': df_preprocessed,
        'drop_cols': drop_cols,
        'encoded_cols': encoded_cols
    })

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

    st.session_state['missing_option'] = missing_option

def handle_preloaded_data(name):
    """
    Handles automatic preprocessing steps for known demo datasets
    """
    #Ensure Clean State
    for key in ['y', 'target_names', 'pca_results', 'visualization_data', 'cluster_labels', 'X_pca']:
        st.session_state.pop(key, None)

    if name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        st.session_state.update({
            'df_raw': df.copy(),
            'X': data.data,
            'y': data.target,
            'target_names': data.target_names.tolist()
        })
    elif name == "Countries":
        df = pd.read_csv("data/Country-data.csv")
        default_target = "country" if "country" in df.columns else df.columns[0]
        
        st.session_state.update({
            'df_raw': df.copy(),
            'X': df.drop(columns=default_target),
            'y': df[default_target],
            'target_names': df[default_target].unique().tolist()
        })

    st.session_state.update({
        'df': df,
        'drop_cols': None,
        'missing_option': None
    })

    st.sidebar.subheader("Select Target Variable")
    default_index = df.columns.get_loc("target" if "target" in df.columns else "country")
    target_col = st.sidebar.selectbox("Select target column", df.columns, index=default_index)
    st.session_state['target_col'] = target_col

def get_model(name, params):
    """
    Returns an initialized unsupervised model based on the model name and parameters.
    """
    if name == "Principal Component Analysis":
        model = PCA(n_components=params["n_components"], random_state=params.get("random_state", None))
    elif name == "K-Means Clustering":
        model = KMeans(n_clusters=params["n_clusters"], random_state=params.get("random_state", None))
    elif name == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params['linkage'])
    return model

def analyze_feature_issues(df, target_col):
    """
    Analyze feature columns for non-numeric types and high missingness.

    Returns:
    - non_numeric_cols: List of non-numeric feature columns
    - high_missing_cols: List of columns with >50% missing values
    """
    feature_cols = df.drop(columns=[target_col])
    non_numeric_cols = feature_cols.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    high_missing_cols = feature_cols.columns[feature_cols.isnull().mean() > 0.5].tolist()
    return non_numeric_cols, high_missing_cols

def apply_feature_cleaning(df, drop_cols, encode_cols):
    """
    Drops and/or one-hot encodes columns in the DataFrame.

    Parameters:
    - drop_cols (list): Columns to drop
    - encode_cols (list): Columns to one-hot encode

    Returns:
    - df_transformed: Cleaned DataFrame
    """
    df_cleaned = df.drop(columns=drop_cols)
    if encode_cols:
        df_encoded = pd.get_dummies(df_cleaned, columns=encode_cols, drop_first=True)
    else:
        df_encoded = df_cleaned
    return df_encoded

def drop_and_encode_features(df, target_col):
    """
    Streamlit UI for guiding the user through dropping or encoding features.
    
    Returns:
    - df_encoded: Transformed DataFrame
    - drop_cols: Columns dropped
    - encode_cols: Columns encoded
    """
    st.sidebar.subheader("Drop or Encode Incompatible Data")

    non_numeric_cols, high_missing_cols = analyze_feature_issues(df, target_col)
    recommended_drop = list(set(non_numeric_cols + high_missing_cols))

    encode_cols = []
    if non_numeric_cols:
        encode_cols = st.sidebar.multiselect(
            "Non-numeric columns to encode (One-Hot):",
            options=non_numeric_cols,
            default=non_numeric_cols
        )

    drop_cols = st.sidebar.multiselect(
        "Columns recommended to drop:",
        options=df.columns.drop(target_col).tolist(),
        default=[col for col in recommended_drop if col not in encode_cols]
    )

    if non_numeric_cols:
        st.sidebar.caption(f"Non-numeric columns: {', '.join(non_numeric_cols)}")
    if high_missing_cols:
        st.sidebar.caption(f"Columns with >50% missing: {', '.join(high_missing_cols)}")

    df_encoded = apply_feature_cleaning(df, drop_cols, encode_cols)

    if encode_cols:
        st.sidebar.success(f"Encoded {len(encode_cols)} column(s) via One-Hot Encoding.")

    return df_encoded, drop_cols, encode_cols

def handle_missing_values(df, missing_option):
    """
    Handle missing values by either dropping rows or imputing numeric columns with the mean.
    Also displays which columns contain missing values.
    """
    df_fixed = df.copy()

    # Identify columns with missing values
    missing_cols = df_fixed.columns[df_fixed.isnull().any()]
    if not missing_cols.empty:
        st.sidebar.markdown("**Columns with missing values**")
        for col in missing_cols:
            st.sidebar.markdown(f"- {col}: {df_fixed[col].isnull().sum()} missing")
    else:
        st.sidebar.success("No missing values detected.")

    # Apply missing value strategy
    if missing_option == "Drop rows":
        df_fixed.dropna(inplace=True)
        st.sidebar.success("Rows with missing values have been dropped.")
    
    elif missing_option == "Impute mean":
        numeric_cols = df_fixed.select_dtypes(include=["float", "int"]).columns
        df_fixed[numeric_cols] = df_fixed[numeric_cols].fillna(df_fixed[numeric_cols].mean())
        st.sidebar.success("Missing numeric values have been imputed with the column mean.")
    
    return df_fixed

def fit_pca(X_scaled, y, selected_features=None, n_components=2):
    """
    This function fits a PCA model on the provided data, ensuring that the features and labels have matching shapes.
    It also handles feature selection if a subset of features is specified.

    Parameters:
    - X_scaled (array-like): The feature matrix (scaled or original).
    - y (array-like): The target labels or classes.
    - selected_features (list, optional): A list of selected feature names to subset the features. Default is None.
    - n_components (int): The number of components to keep in PCA.

    Returns:
    - X_pca (array-like): The transformed data in PCA space.
    - pca (PCA object): The fitted PCA model.
    - y_subset (array-like): The corresponding target labels for the subset used in PCA.
    """

    # Check if selected features are provided
    if selected_features is not None:
        # Ensure selected features are valid column names
        missing_columns = [col for col in selected_features if col not in X_scaled.columns]
        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
            return None, None, None
        X_subset = X_scaled[selected_features]  # Filter `X` based on selected feature names
        y_subset = y  # No filtering on `y`, just ensuring it matches X_subset
    else:
        X_subset = X_scaled  # Use full dataset if no feature selection
        y_subset = y  # Corresponding full target labels

    # Ensure that X and y have the same number of samples
    if len(X_subset) != len(y_subset):
        st.error(f"Feature set (X) and target labels (y) must have the same number of samples. X has {len(X_subset)}, but y has {len(y_subset)}.")
        return None, None, None

    # Apply PCA to the subset (or full data)
    pca = PCA(n_components=n_components, random_state=st.session_state.get("random_state", None))
    X_pca = pca.fit_transform(X_subset)  # Fit PCA and transform the data

    return X_pca, pca, y_subset

def get_pca_projection(X_scaled, y, selected_features=None, n_components=2):
    sf_hash = hash(tuple(selected_features)) if selected_features else "all_features"
    key = f"pca_{n_components}_{sf_hash}"
    
    if key not in st.session_state:
        X_pca, pca, y_subset = fit_pca(X_scaled, y, selected_features, n_components)
        if X_pca is None or pca is None:
            return None, None, None  # Exit early if PCA failed
        st.session_state[key] = (X_pca, pca, y_subset)
    return st.session_state[key]

def plot_pca_projection(X_pca, y, target_names, title="PCA Projection", palette=None):
    """
    Plots a 2D PCA projection colored by labels (numeric or categorical).
    """
    fig, ax = plt.subplots()

    # Basic input checks
    if X_pca is None or y is None or len(X_pca) != len(y):
        st.error("Invalid inputs for PCA projection plot.")
        return None

    # Ensure labels are numeric indices
    le = LabelEncoder()
    y_numeric = le.fit_transform(y)
    class_names = le.classes_ if target_names is None else target_names

    # Color palette fallback
    if palette is None:
        palette = get_color_palette()

    for i, target_name in enumerate(class_names):
        mask = y_numeric == i
        if np.any(mask):  # avoid empty scatter group
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       label=target_name,
                       alpha=0.7, s=60,
                       color=palette[i % len(palette)])

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True)

    # Avoid Plot Overcrowding
    handles, leg_labels = ax.get_legend_handles_labels()
    num_legend_items = len(leg_labels)
    max_num_legend_items = 10
    if num_legend_items > max_num_legend_items:
        st.warning(f"PCA is limited to {max_num_legend_items} different groups. Your data has {num_legend_items}. Try a different plot or adjust parameters to avoid overcrowding.")
        return

    return fig

def plot_pca_projection_with_biplot(X_pca, y, target_names, feature_names, model, show_loadings=False, scaling_factor=50):
    """
    Plots a 2D PCA projection with optional feature vectors (biplot).
    """
    unique_classes = np.unique(y)
    palette = get_color_palette()

    # Avoid Plot Overcrowding
    max_unique_classes = 10
    if len(unique_classes) > max_unique_classes:
        st.warning(f"PCA is limited to {max_unique_classes} different groups. Your data has {len(unique_classes)}. Try a different plot or adjust parameters to avoid overcrowding.")
        return

    fig, ax = plt.subplots()

    # Scatter plot
    for color, class_val, label in zip(palette, unique_classes, target_names):
        ax.scatter(
            X_pca[y == class_val, 0], X_pca[y == class_val, 1],
            color=color, alpha=0.7, label=label,
            s=60
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
    ax.legend(loc="best")
    ax.set_title("PCA Projection with Biplot", fontsize=12)
    ax.grid(True)
    ax.axis('equal')

    return fig

def plot_pca_variance_explained(pca, color1="#66c2a5", color2="#fc8d62"):
    """
    Visualizes the individual and cumulative variance explained by each principal component.
    """
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(explained) + 1)

    fig, ax1 = plt.subplots()
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

def compute_clustering_metrics(X_scaled, method="kmeans", random_state=None, linkage="ward"):
    """
    Compute WCSS (for kmeans) and silhouette scores for k = 2 to 10.
    """
    ks = list(range(2, 11))
    wcss_list = []
    silhouette_scores = []

    for k in ks:
        if method == "kmeans":
            model = KMeans(n_clusters=k, random_state=random_state)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        else:
            raise ValueError("Unsupported method")

        labels = model.fit_predict(X_scaled)

        if method == "kmeans":
            wcss_list.append(model.inertia_)
        else:
            wcss_list.append(None)  # Hierarchical doesn't use WCSS

        silhouette_scores.append(silhouette_score(X_scaled, labels))

    return ks, wcss_list, silhouette_scores

def plot_pca_hierarchical_clustering(X_scaled, cluster_labels, title="Hierarchical Clustering (PCA Projection)", max_clusters=10):
    """
    Streamlit-compatible 2D PCA plot for Hierarchical Clustering.

    Parameters:
    - X_scaled: Scaled feature matrix (NumPy array or DataFrame)
    - cluster_labels: Array of Agglomerative clustering labels
    - title: Plot title
    - max_clusters: Maximum number of clusters to display (default: 10)
    """
    if X_scaled is None or cluster_labels is None:
        st.error("Missing data or clustering labels.")
        return

    # Count unique clusters
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    # Compute 2D PCA projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting
    
    # Define color palette
    palette = get_color_palette()  # returns list of hex colors
    cmap = ListedColormap(palette[:num_clusters])
    
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=cluster_labels,
        s=60,
        alpha=0.7,
        cmap=cmap
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    ax.grid(True)

    # Avoid Plot Overcrowding
    handles, leg_labels = ax.get_legend_handles_labels()
    num_legend_items = len(leg_labels)
    max_num_legend_items = 10
    if num_legend_items > max_num_legend_items:
        st.warning(f"PCA is limited to {max_num_legend_items} different groups. Your data has {num_legend_items}. Try a different plot or adjust parameters to avoid overcrowding.")
        return

    st.pyplot(fig)

st.set_page_config(page_title="DataQuest", layout="wide") # Establish layout

# ==== WELCOME SCREEN ====
if st.session_state.get('welcome_screen', True):
    show_welcome_screen()
    st.stop()

st.title("DataQuest") # Display home page title

# ==== SIDEBAR: DATA UPLOAD & SELECTION ====
st.sidebar.header("Dataset Options")

# Dataset selection
data_source = st.sidebar.selectbox("Choose Dataset", ["Upload CSV", "Breast Cancer", "Countries"])

# Handling user selection for CSV upload
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], 
                                             help="Upload a CSV file containing your dataset. Ensure the file contains a header row with column names.")
    if uploaded_file:
        handle_uploaded_data(uploaded_file)
        st.success("Dataset loaded successfully!")

# Handling in-built datasets
else:
    handle_preloaded_data(data_source)
    st.success(f"{data_source} dataset loaded successfully!")

# === Feature Selection & Training Parameters ===
if 'df' in st.session_state:
    df = st.session_state['df']
    target_col = st.session_state['target_col']

    # Handle Missing Values
    missing_option = st.session_state['missing_option']
    df_fixed = handle_missing_values(df, missing_option)
    st.session_state['df_fixed'] = df_fixed
    st.session_state['y'] = df_fixed[target_col]
    df = st.session_state['df_fixed']

    st.sidebar.subheader("Select Feature Variables")
    feature_candidates = [col for col in df.columns if col != target_col]
    default_features = feature_candidates

    selected_features = st.sidebar.multiselect("Select feature columns", feature_candidates, default=default_features, 
                                               help="Select the columns (features) that will be used to train the model. These should be your input variables (predictors), while the target variable will be excluded from the list of features.")
    if not selected_features:
        st.error("You must select at least one feature.")
        st.stop()

    st.session_state['selected_features'] = selected_features
    st.session_state['X'] = df[selected_features]

    st.session_state['kept_cols'] = selected_features + [target_col]

    st.sidebar.subheader("Set Training Parameters")
    random_state = st.sidebar.number_input("Random state (seed)", value=42, step=1, 
                                           help="The random state (seed) ensures that the results are reproducible. Using the same random state value will produce the same results each time, which is helpful for experiments and debugging.")
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

        # Display Dataframe
        st.write("Preview of processed dataset:")
        kept_cols = st.session_state['kept_cols']
        df = df[kept_cols]
        st.dataframe(df.head())

        X = st.session_state['X']
        y = st.session_state['y']

        # Scale data
        st.subheader("Scale Data")
        st.markdown("Unsupervised methods like PCA or clustering algorithms work better when features are scaled to a similar range, as they are distance-based algorithms.")
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled_array, index=X.index, columns=X.columns) # convert to DataFrame
        st.session_state['X_scaled'] = X_scaled_df  # store as DataFrame
        st.success(f"Data scaled automatically!")
    else:
        st.info("Please select a dataset first.")

# ==== TAB 2: MODEL TRAINING ====
with tab2:
    st.header("Train Model")
    if 'df' in st.session_state:
        df = st.session_state['df']
        X_scaled = st.session_state['X_scaled']

        # Choose a machine learning model and configure parameters
        model_name = st.selectbox("Select model", ["Principal Component Analysis", "K-Means Clustering", "Hierarchical Clustering"],
                                    help="Choose a model for training: PCA is for dimensionality reduction, K-Means is for clustering data, and Hierarchical Clustering is another approach for grouping similar data points.")
        params = {}

        st.session_state["model_name"] = model_name
        
        # If PCA was chosen, display cumulative explained variance
        if model_name == "Principal Component Analysis":

            # Streamlit layout
            st.subheader("Principal Component Analysis")
            st.markdown("PCA reduces data dimensionality while preserving the most significant features (variance). Use the plots below to determine how many components to keep.")
            
            n_samples, n_features = X_scaled.shape
            min_components = min(20, n_features)
            pca_full = PCA(n_components=min_components, random_state=params.get("random_state", None)).fit(X_scaled)
            explained_variance = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            col1, col2 = st.columns(2)

            # Bar Chart
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.bar(range(1, len(explained_variance)+1), explained_variance, color=get_color_palette()[0])
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Explained Variance by Component')
                ax1.set_xticks(range(1, len(explained_variance)+1))
                st.pyplot(fig1)
                st.info("💡 Each bar represents how much variance is explained by the corresponding component.")

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
        
            params["n_components"] = st.slider("Number of Components", 1, min_components, 2, 
                                                help="Number of Components: Choose the number of components to retain. Higher values capture more variance but may reduce interpretability.")
            
        # If KMeans was chosen, output the Elbow plot & Silhouette scores 
        elif model_name == "K-Means Clustering":

            # Streamlit layout
            st.subheader("K-Means Clustering")
            st.markdown("K-Means is a popular unsupervised machine learning algorithm used to partition data into K distinct clusters. It iteratively assigns data points to clusters based on their distance from the cluster centroids.")
            
            # === Calculate WCSS + Silhouette ===

            # Recompute WCSS and silhouette scores only if random state or data has changed
            method_key = f"kmeans_{random_state}_{hash(X_scaled.to_numpy().data.tobytes())}"

            if st.session_state.get("clustering_metrics_key") != method_key:
                ks, wcss_list, silhouette_scores = compute_clustering_metrics(
                    X_scaled, method="kmeans", random_state=random_state
                )
                st.session_state.update({
                    "ks": ks,
                    "wcss_list": wcss_list,
                    "silhouette_scores_list": silhouette_scores,
                    "clustering_metrics_key": method_key
                })
            
            # === Plot Elbow + Silhouette ===
            ks = st.session_state["ks"]
            wcss_list = st.session_state["wcss_list"]
            silhouette_scores = st.session_state["silhouette_scores_list"]

            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(ks, wcss_list, marker='o', color=get_color_palette()[3])
                ax1.set_xlabel("k (Clusters)")
                ax1.set_ylabel("WCSS")
                ax1.set_title("Elbow Method")
                ax1.grid(True)
                st.pyplot(fig1)
                # Explain the elbow method
                st.info("💡 Ideal clustering minimizes the Within-Cluster Sum of Squares (WCSS). Look for the 'elbow' point of sharp decrease to determine an optimal k value.")

            with col2:
                fig2, ax2 = plt.subplots()
                ax2.plot(ks, silhouette_scores, marker='o', color=get_color_palette()[4])
                ax2.set_xlabel("k (Clusters)")
                ax2.set_ylabel("Silhouette Score")
                ax2.set_title("Silhouette Analysis")
                ax2.grid(True)
                st.pyplot(fig2)
                # Explain the silhouette analysis
                st.info("💡 The sillhouette score quantifies how similar an object is to its own cluster compared to other clusters. A silhouette score near +1 means the clusters are well-formed, while a score near -1 suggests poor clustering.")

            params["n_clusters"] = st.slider("Number of Clusters (k)", 2, 10, min(len(np.unique(y)), 10),
                                                    help="Number of clusters (k): Choose a value based on the Elbow and Silhouette analysis. The optimal k typically balances good WCSS and silhouette scores.")
                            
        # If Hierarchical was chosen, output the dendrogram with truncation options
        elif model_name == "Hierarchical Clustering":
            
            # Streamlit Layout - Header and Model Description
            st.subheader("Hierarchical Clustering")
            st.markdown("""
                Hierarchical Clustering is an unsupervised algorithm that builds a tree-like structure (dendrogram) to group 
                data points based on their similarity. It can either be agglomerative (bottom-up) or divisive (top-down), and 
                the number of clusters is determined by cutting the dendrogram at a chosen similarity threshold.
            """)
            
            # Select linkage rule for hierarchical clustering (affects the merging criteria)
            params["linkage"] = st.selectbox("Choose Linkage Rule", ["ward", "single", "complete", "average"],
                                            help="Choose the linkage rule to determine how clusters are merged. 'ward' minimizes variance, 'single' minimizes the closest pair, 'complete' minimizes the farthest pair, 'average' uses average distance.")
            
            linkage_method = params["linkage"]

            # Recompute metrics if parameters or data have changed
            method_key = f"hierarchical_{linkage_method}_{hash(X_scaled.to_numpy().data.tobytes())}"

            if st.session_state.get("clustering_metrics_key") != method_key:
                ks, wcss_list, silhouette_scores = compute_clustering_metrics(
                    X_scaled, method="hierarchical", linkage=linkage_method
                )
                st.session_state.update({
                    "ks": ks,
                    "wcss_list": wcss_list,  # Will be all None
                    "silhouette_scores_list": silhouette_scores,
                    "clustering_metrics_key": method_key
                })
            
            ks = st.session_state["ks"]
            silhouette_scores = st.session_state["silhouette_scores_list"]

            # Silhouette Score Plot
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(ks, silhouette_scores, marker='o', color=get_color_palette()[4])
            ax2.set_xlabel('Number of clusters (k)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title("Silhouette Analysis")
            ax2.grid(True)
            st.pyplot(fig2)
            # Explanation of silhouette score
            st.info("💡 The sillhouette score quantifies how similar an object is to its own cluster compared to other clusters. A silhouette score near +1 means the clusters are well-formed, while a score near -1 suggests poor clustering.")

            # Slider to control the number of clusters to display in the dendrogram (useful for large datasets)
            st.session_state['p_value'] = st.slider("Number of clusters to display in dendrogram (p)",
                                                    min_value=2,
                                                    max_value=max(st.session_state['num_classes'], 20),
                                                    value=max(st.session_state['num_classes'], 20),
                                                    help="Choose the number of clusters to display in the dendrogram. This will limit the number of labels shown for large datasets.")
            
            truncate_mode = "lastp"  # Limit the dendrogram to show only the last 'p' clusters

            # Safe label selection from dataset (handles target or index labels)
            if "target" in df.columns and 'target_names' in st.session_state:
                labels = [st.session_state['target_names'][val] for val in df["target"]]
            elif target_col in df.columns:
                labels = df[target_col].tolist()
            else:
                labels = df.index.astype(str).tolist()  # Use index if no target column exists

            # Perform hierarchical clustering using selected linkage method
            Z = linkage(X_scaled, method=params["linkage"])

            # Dendrogram Plot
            fig1, ax1 = plt.subplots(figsize=(15, 5))
            dendrogram(
                Z,
                labels=labels,  # Using the correct labels for the dendrogram
                truncate_mode=truncate_mode,  # Truncate the dendrogram based on 'p'
                p=st.session_state['p_value']  # Show up to 'p' clusters
            )
            ax1.set_title("Dendrogram", fontsize=16)
            ax1.set_xlabel(target_col, fontsize=14)
            ax1.set_ylabel("Distance", fontsize=14)

            # Rotate x-tick labels to ensure they are readable
            if ax1.get_xticks().size > 0:
                plt.setp(ax1.get_xticklabels(), rotation=90)
                plt.setp(ax1.get_yticklabels(), fontsize=12)

            # Display dendrogram plot
            st.pyplot(fig1)
            st.info("💡 To choose an optimal k value, look for the 'cut-off' point in the dendogram where the vertical lines are longest, indicating the best point to separate clusters.")

            # Slider for the number of clusters (k) for the final cut
            params["n_clusters"] = st.slider("Number of Clusters (k)", 2, 10, 3, 
                                            help="Number of Clusters: Choose the optimal k based on the dendrogram and silhouette score analysis.")
        st.session_state["params"] = params

        # ==== TRAIN THE MODEL ====
        if st.button("Train Model"): # Train model

            params["random_state"] = st.session_state["random_state"]
            model = get_model(model_name, params)
            y = st.session_state['y']
            st.success(f"{model_name} trained successfully!")   
            
            # If PCA was chosen, display cumulative explained variance
            if model_name == "Principal Component Analysis":

                X_pca, pca, _ = get_pca_projection(X_scaled, y, selected_features, params["n_components"])

                explained_var = pca.explained_variance_ratio_
                cum_var = np.cumsum(explained_var)
                total_cum_var = cum_var[-1]  # Last value = total cumulative variance

                # Display Cumulative Explained Variance for the selected number of components
                st.metric("Cumulative Explained Variance", f"{total_cum_var * 100:.2f}%",
                            help="Cumulative explained variance shows the proportion of the dataset's total variance that is captured by each principal component in PCA. It is a running total of the variance explained by the first 'n' components.")

            # If KMeans was chosen
            if model_name == "K-Means Clustering":
                clusters = model.fit_predict(X_scaled)
                st.session_state['clusters'] = clusters 

                current_k = params["n_clusters"]
                ks = st.session_state["ks"]
                sil_scores = st.session_state["silhouette_scores_list"]
                wcss_list = st.session_state["wcss_list"]

                # Lookup correct index
                if current_k in ks:
                    i = ks.index(current_k)
                    sil_score = sil_scores[i]
                    wcss_score = wcss_list[i] if wcss_list[i] is not None else "N/A"
                else:
                    sil_score = silhouette_score(X_scaled, clusters)
                    wcss_score = model.inertia_ if hasattr(model, "inertia_") else "N/A"

                col1, col2 = st.columns(2)

                # Display WCSS and silhouette score for the selected number of clusters
                with col1:
                    st.metric("WCSS", f"{wcss_score:.2f}" if wcss_score != "N/A" else "N/A",
                              help="Within-Cluster Sum of Squares (WCSS) measures the compactness of clusters. It calculates the sum of squared distances between each data point and the centroid of its assigned cluster.")

                with col2:
                    st.metric("Silhouette Score", f"{sil_score:.2f}",
                              help="A silhouette score ranges from -1 to +1. A higher score indicates that points are well-clustered and far from other clusters, meaning the clustering is of good quality. A negative score suggests that points might be assigned to the wrong cluster.")   
                                
            # If Hierarchical was chosen
            if model_name == "Hierarchical Clustering":
                df["Cluster"] = model.fit_predict(X_scaled)
                st.session_state['df']['Cluster'] = df["Cluster"]
                cluster_labels = df["Cluster"].tolist()
                st.session_state['cluster_labels'] = cluster_labels   

                current_k = params["n_clusters"]
                ks = st.session_state["ks"]
                sil_scores = st.session_state["silhouette_scores_list"]

                # Lookup correct index
                if current_k in ks:
                    i = ks.index(current_k)
                    sil_score = sil_scores[i]
                else:
                    sil_score = silhouette_score(X_scaled, df["Cluster"])

                # Display silhouette score for the selected number of clusters
                st.metric("Silhouette Score", f"{sil_score:.2f}",
                          help="A silhouette score ranges from -1 to +1. A higher score indicates that points are well-clustered and far from other clusters, meaning the clustering is of good quality. A negative score suggests that points might be assigned to the wrong cluster.")         
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
        
        st.subheader("PCA Projection")
        
        X_scaled = st.session_state["X_scaled"]
        y = st.session_state["y"]
        selected_features = st.session_state.get("selected_features", [])
        n_components = st.session_state["params"]["n_components"]

        X_pca, pca, _ = get_pca_projection(X_scaled, y, selected_features, n_components)
        if X_pca is None or pca is None:
            st.error("Unable to compute PCA projection for visualization.")
            st.stop()

        label_names = st.session_state.get("target_names", [str(i) for i in np.unique(y)])

        # Toggle biplot
        show_loadings = st.checkbox(
            "Show loadings (biplot)",
            value=False,
            help="Toggle to display PCA loadings as arrows on the biplot."
        )

        if show_loadings:
            scaling_factor = st.slider(
            "Loading arrow scale",
            min_value=1,
            max_value=100,
            value=50,
            help="Adjusts the length of the loading arrows in the biplot to improve visibility."
        )
            fig = plot_pca_projection_with_biplot(X_pca, y, label_names, selected_features, pca,
                                                show_loadings=True, scaling_factor=scaling_factor)
        else:
            fig = plot_pca_projection(X_pca, y, label_names)
    
        if fig:
            st.pyplot(fig)

        # Also show variance explained
        st.subheader("Variance Explained by Principal Components")
        fig = plot_pca_variance_explained(pca)
        st.pyplot(fig)
        st.info("💡 Look for the 'elbow' in the cumulative curve to determine an optimal number of components.")
    
    # ==== K-Means Clustering ====  
    if st.session_state.get("model_name") == "K-Means Clustering":
        st.subheader("2D Projection of Clusters (via PCA)")
        
        X_scaled = st.session_state.get("X_scaled")
        y = st.session_state.get("y")
        selected_features = st.session_state.get("selected_features", [])
        n_clusters = st.session_state.get("params", {}).get("n_clusters", 2)
        random_state = st.session_state.get("random_state", 42)
        color_mode = st.selectbox("Color points by", ["K-Means Clusters", "Target Labels"],
                                help="Choose to color the points by their K-Means cluster label or by the original target labels.")
        
        # === Get or fit KMeans model ===
        current_key = f"kmeans_{n_clusters}_{random_state}"
        if st.session_state.get("kmeans_key") != current_key:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            clusters = kmeans.fit_predict(X_scaled)
            st.session_state.update({
                "kmeans": kmeans,
                "clusters": clusters,
                "kmeans_key": current_key
            })
        else:
            clusters = st.session_state["clusters"]
        
        # === Set labels and label names for plotting ===
        if color_mode == "K-Means Clusters":
            labels = clusters
            label_names = [f"Cluster {i}" for i in np.unique(labels)]
        else:
            labels = y
            label_names = st.session_state.get("target_names", [str(i) for i in np.unique(y)])
        
        # === Get 2D PCA projection (cached if available) ===
        X_pca, pca, _ = get_pca_projection(X_scaled, y, selected_features, 2)
        if X_pca is None or pca is None:
            st.error("Unable to compute PCA projection for visualization.")
            st.stop()

        # === Plot PCA projection ===
        palette = get_color_palette()
        fig = plot_pca_projection(X_pca, labels, label_names,
                                f"K-Means Clustering (PCA Projection Colored by {color_mode})", palette)
        if fig:
            st.pyplot(fig)
        
    # ==== Hierarchical Clustering ====  
    if st.session_state.get("model_name") == "Hierarchical Clustering":
        if "cluster_labels" in st.session_state and "X_scaled" in st.session_state:
            X_scaled = st.session_state["X_scaled"]
            cluster_labels = st.session_state["cluster_labels"]

            # Plot the hierarchical clustering result
            plot_pca_hierarchical_clustering(X_scaled, cluster_labels)
        else:
            st.warning("Cluster labels or scaled data not found. Please ensure clustering is performed first.")