# :robot: DataFlex
*An Interactive Machine Learning App*  

## Project Overview  
**DataFlex** is a [**Streamlit**](https://streamlit.io/) app that applies supervised machine learning (ML) models to any dataset. You can use DataFlex to make predictions, adjust hyperparameters, and evaluate model performance in parallel.

### Features
**Make predictions**: DataFlex is equipped for the following supervised machine learning models:
- Linear Regression
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)<br><br>
**Adjust hyperparameters**: Model-specific hyperparameter sliders allow you to balance model performance against processing time and simplicity by tuning values such as:
- Depth
- Max Split
- etc.<br><br>
**Evaluate Model Performance**: Assess and compare various models by comparing numerous evaluation metrics:
- Accuracy
- Precision
- Recall/Sensitivity
- Specificity
- Gini Index

## Getting Started

### Install Dependencies  
Install & import the necessary libraries with the following commands:
```bash
pip install streamlit pandas seaborn matplotlib
```
```python
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
### Run the App
Access the [**MLStreamlit**](https://mccartan-mlstreamlit-app.streamlit.app/) app on any device!

## Using the App

### Choose your Data  
Choose from 3 demo datasets to practice making predictions, tuning hyperparameters, and evaluating models!
- **Demo Dataset 1: ____**: Explanation of dataset
- **Demo Dataset 2: ____**: Explanation of dataset
- **Demo Dataset 3: ____**: Explanation of dataset<br>

Or, use the 'input' tab to upload your own .csv file

