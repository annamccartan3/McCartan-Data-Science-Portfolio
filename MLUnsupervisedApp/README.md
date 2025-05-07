# 💡 DataQuest  
*An Unsupervised Machine Learning Explorer*

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [References](#references)

---

## Project Overview  
**DataQuest** is an interactive **Streamlit** app that allows users to explore unsupervised machine learning techniques like clustering and dimensionality reduction.

### What is supervised machine learning?  
Unsupervised learning refers to techniques that discover patterns in data without using labeled outcomes. As described in *Grokking Machine Learning* by **Luis G. Serrano**, these methods help reveal structure, groupings, and key relationships in datasets. Examples include:
- Dimensionality Reduction – Simplifying high-dimensional data while preserving key variance (e.g., PCA)
- Clustering – Grouping data points based on similarity (e.g., K-Means, Agglomerative)

Whether you're analyzing hidden structures in your data or exploring high-dimensional datasets, DataQuest offers hands-on tools to apply methods like Principal Component Analysis (PCA), and K-Means and Hierarchical Clustering.

---

## Getting Started

### Deployed App
Access the live version here: [**Deployed App on Streamlit Cloud**](https://dataquest.streamlit.app/)

### Run the App Locally

#### Clone & Navigate to the Repository
```bash
git clone https://github.com/annamccartan3/McCartan-Data-Science-Portfolio.git
cd McCartan-Data-Science-Portfolio/MLUnsupervisedApp
```

#### Install Dependencies  
All dependencies can be found in the requirements.txt file. Create and activate a virtual environment, then install the necessary libraries:
```bash
pip install -r requirements.txt
```

#### Run the Streamlit App
```bash
streamlit run main.py
```

Once you’re in the app, follow these steps to begin your machine learning workflow:
1. Upload your dataset or choose a demo dataset.
2. Select features and a target variable for prediction.
3. Train a model, adjust hyperparameters, and evaluate performance.
4. Explore visualizations to understand how your model is performing.

---

## Datasets  
*Choose from three built-in demo datasets or upload your own `.csv` file.*<br>
#### Demo Datasets:
- **[Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** – Predict tumor malignancy from medical imaging features.
- **[Countries]([https://en.wikipedia.org/wiki/Passengers_of_the_Titanic](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data))** – Explore demographic differences in health and economic indicators across the globe.

#### DataQuest includes built-in tools for:
- **Dropping or Encoding incompatible data**: Automatically detect and encode categorical columns, or drop non-numeric/identifier columns that could interfere with model training.
- **Handling missing values**: If missing data is found, choose to drop affected rows or impute missing values with the mean.
- **Running Reproducible Tests**: Set a random state for reproducibility in model evaluation and visualization over time.

---

## Model Training  
Once features and target variables are selected, choose from the following machine learning models:
- **Principal Component Analysis (PCA)** – A dimensionality reduction technique that transforms features into a set of uncorrelated components for easier visualization and analysis.
- **K-Means Clustering** –  An unsupervised learning algorithm that partitions data into k clusters based on feature similarity.
- **Hierarchical Clustering** –  An unsupervised algorithm that builds a hierarchy of clusters using pairwise distances between data points.

Additional options include:
- Modifying the chosen target and feature variables
- Tuning hyperparameters (e.g., number of components, number of clusters, linkage type)

---

## Evaluation Metrics  
To assess your model, DataQuest provides key metrics:
- **Principal Component Analysis (PCA): Explained Variance**
  - Explained Variance -
  - Cumulative Explained Variance -
- **K-Means Clustering: Explained Variance**
  - WCSS
  - Silhouette Score
- **Hierarchical Clustering: Explained Variance**
  - Silhouette Score

These metrics are displayed after training to help you compare model effectiveness.

---

## Visualization  
Visual tools help bring your model results to life:
- **Feature Importance Plot** – Identifies which features have the most influence in logistic regression
<img src="images/FeatureImportance.png" height="300">

- **Decision Tree Graph** – Shows the structure and decisions of a tree-based model
<img src="images/DecisionTree.png" height="300">

- **KNN k-value Analysis** – Plots accuracy and F1-score for various values of *k* to help tune your model
<img src="images/KNN.png" height="300">

- **Confusion Matrix** – Displays counts of correct (TP/TN, along diagonal) vs. incorrect (FP/FN) predictions by class
<img src="images/CM.png" height="300">

- **ROC Curve** – Shows how well your model can distinguish between positive and negative cases
  - **Area Under the Curve (AUC)** - Measures how successful your model is at making this distinction
<img src="images/ROC.png" height="300">

---

## References  

### Machine Learning  
- *Grokking Machine Learning* by Luis G. Serrano  

### Demo Datasets  
- [Titanic Dataset](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic) – via Seaborn  
- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) – via Scikit-learn  
- [Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) – via Scikit-learn 
