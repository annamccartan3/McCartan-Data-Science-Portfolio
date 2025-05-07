# dataquest_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

# Set global styles
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("pastel")

st.set_page_config(page_title="DataQuest: Unsupervised Learning Explorer", layout="wide")
st.title("🔍 DataQuest: Explore Unsupervised Learning Models")

# ==== SIDEBAR: DATA UPLOAD & SELECTION ====
st.sidebar.header("📁 Upload or Load Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using example Iris dataset.")
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

st.subheader("📊 Raw Dataset Preview")
st.dataframe(df.head())

# ==== DATA PREPROCESSING ====
st.subheader("⚙️ Preprocessing")
features = st.multiselect("Select features for clustering:", df.columns.tolist(), default=df.columns.tolist())
data = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# ==== SIDEBAR: MODEL SELECTION ====
st.sidebar.header("⚙️ Model Settings")
model_type = st.sidebar.selectbox("Choose Clustering Model", ["K-Means", "Hierarchical", "DBSCAN"], help="Select a clustering model to apply.")

# ==== K-MEANS CLUSTERING ====
if model_type == "K-Means":
    st.subheader("📌 K-Means Clustering")
    k = st.sidebar.slider("Select number of clusters (k):", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    st.metric("Silhouette Score", f"{silhouette:.3f}")
    df["Cluster"] = labels

    # Plot Clusters in PCA Space
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2", s=60)
    ax.set_title("Clusters (PCA Projection)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

# ==== HIERARCHICAL CLUSTERING ====
elif model_type == "Hierarchical":
    st.subheader("🧬 Hierarchical Clustering")
    method = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average", "single"])
    dendro_cut = st.sidebar.slider("Number of clusters (cut dendrogram at):", 2, 10, 3)

    linked = linkage(X_scaled, method=method)
    labels = fcluster(linked, t=dendro_cut, criterion='maxclust')
    silhouette = silhouette_score(X_scaled, labels)

    st.metric("Silhouette Score", f"{silhouette:.3f}")
    df["Cluster"] = labels

    # Dendrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode="lastp", p=30, ax=ax)
    ax.set_title("Hierarchical Dendrogram")
    st.pyplot(fig)

    # PCA Cluster Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2", s=60)
    ax.set_title("Clusters (PCA Projection)")
    st.pyplot(fig)

# ==== DBSCAN CLUSTERING ====
elif model_type == "DBSCAN":
    st.subheader("📍 DBSCAN Clustering")
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("Minimum samples", 1, 10, 5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, labels)
        st.metric("Silhouette Score", f"{silhouette:.3f}")
    else:
        st.warning("Could not form more than one cluster.")

    df["Cluster"] = labels

    # PCA Cluster Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2", s=60)
    ax.set_title("Clusters (PCA Projection)")
    st.pyplot(fig)