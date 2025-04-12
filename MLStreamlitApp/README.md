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
**DataFlex** is an interactive [**Streamlit**](https://streamlit.io/) app that applies supervised machine learning (ML) models to any dataset. Use DataFlex to make predictions, tune hyperparameters, and visualize model performance with ease.

### What is supervised machine learning?  
Supervised machine learning, as described by **Luis G. Serrano** in *Grokking Machine Learning*, is the process of training a model using labeled data to make accurate predictions. This typically involves:
- Splitting the data into training and testing sets  
- Instantiating and training a model on the training data  
- Using the model to make predictions on unseen test data  
- Evaluating the model's performance by comparing predictions with actual outcomes  

With DataFlex, you can experiment with different models—including Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN)—and observe how preprocessing steps and parameter tuning affect the results.

---

## Getting Started  
👉 [**Launch DataFlex App**](https://mccartan-mlstreamlit-app.streamlit.app/)

Once you’re in the app, follow these steps to begin your machine learning workflow:
1. Upload your dataset or choose a demo dataset.
2. Select features and a target variable for prediction.
3. Train a model, adjust hyperparameters, and evaluate performance.
4. Explore visualizations to understand how your model is performing.

---

## Datasets  
Choose from three built-in demo datasets or upload your own `.csv` file:

- **[Titanic](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic)** – Explore survival patterns based on passenger attributes.
- **[Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set)** – Classify iris flower species using petal and sepal measurements.
- **[Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** – Predict tumor malignancy from medical imaging features.

If you upload a custom dataset, DataFlex includes built-in tools for:
- Handling missing values (drop or impute)
- Scaling features
- Customizing train-test splits

---

## Model Training  
Once features and target variables are selected, choose from the following machine learning models:
- **Logistic Regression** – Great for binary classification problems.
- **Decision Tree** – Interpretable tree-based model that handles both classification and regression.
- **K-Nearest Neighbors (KNN)** – Classifies data points based on proximity in feature space.

Additional options include:
- Scaling data using standardization or min-max normalization  
- Tuning hyperparameters (e.g., regularization strength, tree depth, or number of neighbors)  
- Adjusting train-test split ratios and random state for reproducibility  

---

## Evaluation Metrics  
To assess your model, DataFlex provides key metrics:
- **Accuracy** – Proportion of correct predictions  
- **Precision** – Proportion of positive predictions that were actually correct  
- **Recall** – Proportion of actual positives that were correctly predicted  
- **F1 Score** – Harmonic mean of precision and recall  
- **ROC AUC Score** – Measures classification performance across thresholds (for binary tasks)

These metrics are displayed after training to help you compare model effectiveness.

---

## Visualization  
Visual tools help bring your model results to life:
- 📊 **Confusion Matrix** – Highlights correct and incorrect predictions  
- 📈 **ROC Curve** – Visualizes true positive rate vs. false positive rate  
- 🌿 **Decision Tree Graph** – Shows the structure and decisions of a tree-based model  
- 📌 **Feature Importance Plot** – Identifies which features have the most influence  
- 🔍 **KNN k-value Analysis** – Plots accuracy and F1-score for various values of *k* to help tune your model  

---

## References  

### Machine Learning  
- *Grokking Machine Learning* by Luis G. Serrano  

### Demo Datasets  
- [Titanic Dataset](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic) – via Seaborn  
- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) – via Scikit-learn  
- [Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) – via Scikit-learn 
