# 🤖 DataFlex  
*An Interactive Machine Learning App*

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
**DataFlex** is an interactive [**Streamlit**](https://streamlit.io/) app that applies supervised machine learning (ML) classification models to any dataset. Use DataFlex to make predictions, tune hyperparameters, and visualize model performance.

### What is supervised machine learning?  
Supervised machine learning, as described by **Luis G. Serrano** in *Grokking Machine Learning*, is the process of training a model using labeled data to make accurate predictions. This typically involves:
- Splitting the data into training and testing sets  
- Instantiating and training a model on the training data  
- Using the model to make predictions on unseen test data  
- Evaluating the model's performance by comparing predictions with actual outcomes  

With DataFlex, you can experiment with different classification models—including Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN)—and observe how preprocessing steps and parameter tuning affect the results.

---

## Getting Started

### Deployed App
Access the live version here: [**Deployed App on Streamlit Cloud**](https://mccartan-mlapp.streamlit.app/)

### Run the App Locally

#### Clone the Repository
```bash
git clone https://github.com/annamccartan3/McCartan-Data-Science-Portfolio.git
cd McCartan-Data-Science-Portfolio/MLStreamlitApp
```

#### Install Dependencies  
All dependencies can be found in the requirements.txt file. It is recommended to create and activate a virtual environment before installing the necessary libraries with the following command:
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
- **[Titanic](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic)** – Explore survival patterns based on passenger attributes.
- **[Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set)** – Classify iris flower species using petal and sepal measurements.
- **[Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** – Predict tumor malignancy from medical imaging features.

#### DataFlex includes built-in tools for:
- **Customizing train-test splits**: Define your training/testing ratio and set a random state for reproducibility.
- **Dropping incompatible data**: Automatically detect and drop non-numeric/identifier columns that could interfere with model training.
- **Handling missing values**: If missing data is found, choose to drop affected rows or impute missing values with the mean.
  
---

## Model Training  
Once features and target variables are selected, choose from the following machine learning models:
- **Logistic Regression** – Great for binary classification problems.
- **Decision Tree** – Interpretable tree-based model that breaks classification into a series of binary steps.
- **K-Nearest Neighbors (KNN)** – Classifies data points based on their similarity to each other.

Additional options include:
- Modifying the chosen target and feature variables
- Scaling numeric data using StandardScaler
- Tuning hyperparameters (e.g., tree depth, number of neighbors)  

---

## Evaluation Metrics  
To assess your model, DataFlex provides key metrics:
- **Accuracy** – Proportion of correct predictions  
- **Precision** – Proportion of positive predictions that were actually correct  
- **Recall** – Proportion of actual positives that were correctly predicted  
- **F1 Score** – Harmonic mean of precision and recall

These metrics are displayed after training to help you compare model effectiveness.

---

## Visualization  
Visual tools help bring your model results to life:
- **Feature Importance Plot** – Identifies which features have the most influence in logistic regression
- **Decision Tree Graph** – Shows the structure and decisions of a tree-based model
- **KNN k-value Analysis** – Plots accuracy and F1-score for various values of *k* to help tune your model  
- **Confusion Matrix** – Displays counts of correct (TP/TN, along diagonal) vs. incorrect (FP/FN) predictions by class
- **ROC Curve** – Shows how well your model can distinguish between positive and negative cases
  - **Area Under the Curve (AUC)** - Measures how successful your model is at making this distinction

---

## References  

### Machine Learning  
- *Grokking Machine Learning* by Luis G. Serrano  

### Demo Datasets  
- [Titanic Dataset](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic) – via Seaborn  
- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) – via Scikit-learn  
- [Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) – via Scikit-learn 
