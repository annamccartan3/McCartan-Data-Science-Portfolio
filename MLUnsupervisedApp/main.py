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
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# add 1 more demo??
# PCA for >> 2 classes ? 
# Color code graphs, consistent color scheme/font sizing
# add tooltips and helpful user advice
# organize code
# add thorough comments and organization

# Helper Functions
def load_data(file):
    df = pd.read_csv(file)
    return df

def get_model(name, params):
    if name == "Principal Component Analysis":
        model = PCA(n_components=params["n_components"], random_state=st.session_state['random_state'])
    elif name == "K-Means Clustering":
        model = KMeans(n_clusters=params["n_clusters"], random_state=st.session_state['random_state'])
    elif name == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="ward")
    return model

def drop_and_encode_features(df, target_col):
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

# Streamlit Interface
if 'welcome_screen' not in st.session_state:
    st.session_state['welcome_screen'] = True

if st.session_state['welcome_screen']:
    st.title("Welcome to DataQuest!")
    st.write("DataQuest allows you to upload datasets, group data with unsupervised machine learning models, and visualize the results.")
    st.write("Follow these steps to get started:")
    st.write("1. Upload a dataset or select a demo dataset.")
    st.write("2. Choose features and the target variable.")
    st.write("3. Train a model and tune hyperparameters.")
    st.write("4. Visualize model performance.")
    st.button("Get Started", on_click=lambda: st.session_state.update({'welcome_screen': False}))
else:
    st.set_page_config(page_title="DataQuest", layout="wide")
    st.title("DataQuest")

    # Sidebar for controls
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

    # Tab 1: Refine Data
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

    # Tab 2: Train Model
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
                        ax1.bar(range(1, len(explained_variance)+1), explained_variance, color='skyblue')
                        ax1.set_xlabel('Principal Component')
                        ax1.set_ylabel('Explained Variance Ratio')
                        ax1.set_title('Explained Variance by Component')
                        ax1.set_xticks(range(1, len(explained_variance)+1))
                        st.pyplot(fig1)

                    # Cumulative Variance Plot
                    with col2:
                        fig2, ax2 = plt.subplots()
                        ax2.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--', color='green')
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
                            ax1.plot(ks, wcss, marker='o')
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
                            ax2.plot(ks, silhouette_scores, marker='o', color='green')
                            ax2.set_xlabel('Number of clusters (k)')
                            ax2.set_ylabel('Silhouette Score')
                            ax2.set_title('Silhouette Analysis')
                            ax2.grid(True)
                            st.pyplot(fig2)
                            # Explain the purpose of this plot
                            st.info("💡 The sillhouette score quantifies how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering.")
                    
                        params["n_clusters"] = st.slider("Number of Clusters (k)", 1, 10, 2, help="Number of Clusters: [insert description here]")
                    
                    else:
                        st.warning("Please compute WCSS and Silhouette Scores before visualizing.")
                
                # If Hierarchical was chosen, output the dendogram w/ truncation options
                elif model_name == "Hierarchical Clustering":

                    ks = list(range(2, 11))
                    silhouette_scores = []

                    for k in ks:
                        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
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
                        ax2.plot(ks, silhouette_scores, marker='o', color='green')
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
                    Z = linkage(X_scaled, method="ward")
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
                    random_state = st.session_state.get('random_state', 42)
                    st.success(f"{model_name} trained successfully!")   
                    
                    # If PCA was chosen, display cumulative explained variance
                    if model_name == "Principal Component Analysis":
                        X_pca = model.fit_transform(X_scaled)
                        st.session_state['X_pca'] = X_pca 
                    
                    # If KMeans was chosen
                    if model_name == "K-Means Clustering":
                        clusters = model.fit_predict(X_scaled)
                        st.session_state['clusters'] = clusters 
                        kmeans_accuracy = accuracy_score(y, clusters)
                        st.metric("Accuracy", f"{kmeans_accuracy*100:.2f}%")
                    
                    # If Hierarchical was chosen
                    if model_name == "Hierarchical Clustering":
                        df["Cluster"] = model.fit_predict(X_scaled)
                        st.session_state['df']['Cluster'] = df["Cluster"]
                        cluster_labels = df["Cluster"].tolist()
                        st.session_state['cluster_labels'] = cluster_labels            

            else:
                st.info("To train a model, please select a dataset first.")

    # Tab 3: Visualization
        with tab3:
            st.header("Visualization")
            if "model_name" in st.session_state:
                st.subheader(model_name)
                if st.session_state["model_name"] == "Principal Component Analysis":

                    st.subheader("2D Projection")
                    
                    # Ensure PCA data and labels are available
                    if "X_pca" in st.session_state and "y" in st.session_state:
                        X_pca = st.session_state["X_pca"]
                        y = st.session_state["y"]
                        unique_classes = np.unique(y)
                        num_classes = st.session_state['num_classes']

                        # Limit number of classes for visual clarity
                        MAX_CLASSES = 10
                        if num_classes > MAX_CLASSES:
                            st.warning(f"PCA plot is only supported for up to {MAX_CLASSES} classes. Your data has {num_classes}.")
                        else:
                            # Get target names if available
                            target_names = st.session_state.get("target_names", [str(label) for label in unique_classes])

                            # Color palette adapts to number of classes
                            palette = sns.color_palette("hls", num_classes)

                            # Begin plotting
                            plt.figure(figsize=(8, 6))
                            for color, class_val, label in zip(palette, unique_classes, target_names):
                                plt.scatter(
                                    X_pca[y == class_val, 0],
                                    X_pca[y == class_val, 1],
                                    color=color, alpha=0.7, label=label,
                                    edgecolor='k', s=60
                                )

                            plt.xlabel("Principal Component 1")
                            plt.ylabel("Principal Component 2")
                            plt.title("2D Projection of Dataset")
                            plt.legend(loc="best")
                            plt.grid(True)
                            plt.axis('equal')  # Ensures arrows are proportional

                            # Option to show PCA feature loadings
                            show_loadings = st.checkbox("Show PCA Feature Loadings (Biplot Arrows)")
                            if show_loadings:
                                # Slider for scaling factor
                                scaling_factor = st.slider("Adjust arrow scaling", 1.0, 100.0, 50.0, step=1.0)

                                model = st.session_state.get("model")
                                feature_names = st.session_state.get("selected_features", [])

                                if model and hasattr(model, "components_"):
                                    loadings = model.components_.T  # Shape: (n_features, n_components)

                                    # Check for shape mismatch
                                    if len(feature_names) != loadings.shape[0]:
                                        st.error("Mismatch between selected features and PCA loadings.")
                                    else:
                                        for i, feature in enumerate(feature_names):
                                            x_loading = scaling_factor * loadings[i, 0]
                                            y_loading = scaling_factor * loadings[i, 1]

                                            plt.arrow(0, 0, x_loading, y_loading,
                                                    color='red', width=0.005, head_width=0.1, alpha=0.7, length_includes_head=True)

                                            plt.text(x_loading * 1.15, y_loading * 1.15, feature,
                                                    color='darkred', fontsize=9, ha='center', va='center')
                                else:
                                    st.error("PCA model not found or not fitted.")

                            # Show the plot
                            st.pyplot(plt.gcf())

                    else:
                        st.warning("Please train the PCA model in the 'Train PCA' tab first.")
                                    
                    # Example: Fit PCA and prepare data
                    pca = PCA(n_components=params["n_components"])
                    pca.fit(X_scaled)
                    explained = pca.explained_variance_ratio_ * 100  # Convert to percentage
                    cumulative = np.cumsum(explained)
                    components = np.arange(1, len(explained) + 1)

                    # Create the combined plot
                    fig, ax1 = plt.subplots(figsize=(8, 6))

                    # Bar plot for individual variance explained
                    bar_color = 'steelblue'
                    ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
                    ax1.set_xlabel('Principal Component')
                    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
                    ax1.tick_params(axis='y', labelcolor=bar_color)
                    ax1.set_xticks(components)
                    ax1.set_xticklabels([f"PC{i}" for i in components])

                    # Add percentage labels on each bar
                    for i, v in enumerate(explained):
                        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

                    # Line plot for cumulative variance explained (on secondary y-axis)
                    ax2 = ax1.twinx()
                    line_color = 'crimson'
                    ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
                    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
                    ax2.tick_params(axis='y', labelcolor=line_color)
                    ax2.set_ylim(0, 100)

                    # Combine legends from both axes
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

                    plt.title('PCA: Variance Explained', pad=20)
                    plt.tight_layout()

                    # Display in Streamlit
                    st.subheader("Explained Variance")
                    st.pyplot(fig)

                    # Graphing tip
                    st.info("💡 Look for the 'elbow' in the cumulative curve to determine an optimal number of components.")
            
                if st.session_state.get("model_name") == "K-Means Clustering":
                    
                    if "X_scaled" in st.session_state:
                        X_scaled = st.session_state["X_scaled"]
                        n_clusters = st.session_state.get("params", {}).get("n_clusters", 2)

                        color_mode = st.selectbox("Color points by", ["K-Means Clusters", "Target Labels"])

                        if color_mode == "K-Means Clusters":
                            # If not stored or cluster count doesn't match, recompute
                            if "kmeans" not in st.session_state or \
                            st.session_state.get("params", {}).get("n_clusters", None) != n_clusters:
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                clusters = kmeans.fit_predict(X_scaled)
                                st.session_state["kmeans"] = kmeans
                                st.session_state["clusters"] = clusters
                                st.session_state["params"] = {"n_clusters": n_clusters}
                            else:
                                kmeans = st.session_state["kmeans"]
                                clusters = st.session_state["clusters"]

                            color_labels = clusters
                            label_names = [f"Cluster {i}" for i in np.unique(clusters)]
                        else:
                            if "y" not in st.session_state:
                                st.warning("Target variable not found in session. Please load or define a target.")
                                st.stop()
                            y = st.session_state["y"]
                            color_labels = y
                            labels = np.unique(y)
                            label_names = (
                                st.session_state.get("target_names", [str(i) for i in labels])
                                if len(labels) <= 10 else [f"Class {i}" for i in labels]
                            )

                        # PCA Projection (fit & transform only once)
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        st.session_state["pca"] = pca

                        # Plotting
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels_unique = np.unique(color_labels)
                        cmap = cm.get_cmap("tab10", len(labels_unique))

                        for i, label in enumerate(labels_unique):
                            ax.scatter(
                                X_pca[color_labels == label, 0],
                                X_pca[color_labels == label, 1],
                                color=cmap(i),
                                label=label_names[i],
                                edgecolor="k",
                                alpha=0.7,
                                s=60,
                            )

                        ax.set_xlabel("Principal Component 1")
                        ax.set_ylabel("Principal Component 2")
                        ax.set_title(f"PCA Projection Colored by {color_mode}")
                        ax.legend(loc="best")
                        ax.grid(True)
                        st.pyplot(fig)
                
                if st.session_state.get("model_name") == "Hierarchical Clustering":
                    
                    if "cluster_labels" in st.session_state and "X_scaled" in st.session_state:

                        X_scaled = st.session_state["X_scaled"]
                        cluster_labels = st.session_state["cluster_labels"]

                        # PCA Projection (fit & transform only once)
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        st.session_state["pca"] = pca

                        # Plotting
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels_unique = np.unique(cluster_labels)
                        cmap = cm.get_cmap("tab10", len(labels_unique))

                        for i, label in enumerate(labels_unique):
                            idx = np.where(np.array(cluster_labels) == label)
                            ax.scatter(
                                X_pca[idx, 0],
                                X_pca[idx, 1],
                                label=f"Cluster {label}",
                                c=np.array([cmap(i)]),
                                edgecolor="k",
                                alpha=0.7,
                                s=60,
                            )

                        ax.set_xlabel("Principal Component 1", fontsize=12)
                        ax.set_ylabel("Principal Component 2", fontsize=12)
                        ax.set_title("Agglomerative Clustering on Data (via PCA)", fontsize=14)
                        ax.legend(loc="best")
                        ax.grid(True)
                        st.pyplot(fig)

            else:
                st.info("To view visualization, train a model in Tab 2 first.")