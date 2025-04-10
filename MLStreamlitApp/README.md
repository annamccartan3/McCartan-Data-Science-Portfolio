# :robot: DataFlex
*An Interactive Machine Learning App*  

## Project Overview  
**DataFlex** is a [**Streamlit**](https://streamlit.io/) app that applies supervised machine learning (ML) models to any dataset. You can use DataFlex to make predictions, adjust hyperparameters, and visualize model performance.

### What is supervised machine learning?  
Supervised machine learning, as outlined by **Luis G. Serrano** in *Grokking Machine Learning*, is a process of making predictions based on labeled training data through the following process:  
- Data is split into training and testing sets
- A model is instantiated and trained
- The model makes predictions on the test data
- Model performance is evaluated based on how closely the predictions match the actual labels.

Depending on the dataset, many different models can be used to make predictions, including Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN). With DataFlex, you can explore all of these models and see how changing the model type, scaling data, and tuning hyperparameters influences model performance.
__

## Getting Started
Click [**here**](https://mccartan-mlstreamlit-app.streamlit.app/) to access DataFlex on any device! Once you're in, follow these steps to get started:
1. Upload a dataset or select a demo dataset.
2. Choose features and the target variable.
3. Train a model, tune hyperparameters, and view evaluation metrics.
4. Visualize model performance.

## Datasets
Choose from 3 demo datasets to practice supervised machine learning in multiple contexts:
- The [**Titanic**](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic) dataset allows you to explore the factors that influenced passenger survival rates on the Titanic.
- The [**Iris**](https://en.wikipedia.org/wiki/Iris_flower_data_set) dataset contains measurements for the morphological features of three different species of *Iris* flowers, introducing the potential for multiclass identification. 
- The [**Breast Cancer**](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) dataset describes characteristics of various images of breast tissue, demonstrating the diagnostic potential of machine learning algorithms. <br>

Or, upload your own data as a .csv file to make new predictions!

## Model Training

## Evaluation Metrics

## Visualization

__

## References
### Machine Learning
- [*Grokking Machine Learning*] by Luis G. Serrano
### Demo Datasets
- [Titanic](https://en.wikipedia.org/wiki/Passengers_of_the_Titanic), loaded from Seaborn
- [Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set), loaded from Scikit-learn
- [Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded from Scikit-learn
