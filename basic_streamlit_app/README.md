# :dog: DogFinder :poodle:
*A Streamlit App for Dog Data Analysis*

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Dogfinder Dataset](#dogfinder-dataset)
- [App Features](#app-features)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [References](#references)

---

## Project Overview  
**DogFinder** is an interactive **Streamlit** app designed to track information across a range of dog breeds. The project enables interactive visualization of characteristics such as size, behavior, and friendliness.

---

## Getting Started

### Deployed App
Access the live version here: [**Deployed App on Streamlit Cloud**](https://dogfinder.streamlit.app/)

### Run the App Locally

#### Clone & Navigate to the Repository
```bash
git clone https://github.com/annamccartan3/McCartan-Data-Science-Portfolio.git
cd McCartan-Data-Science-Portfolio/basic_streamlit_app
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

Once you’re in the app, follow these steps to begin:
1. [insert]

---

## DogFinder Dataset
DogFinder utilizes a cleaned dataset from [Kaggle](https://www.kaggle.com/datasets/yonkotoshiro/dogs-breeds), which contains information on nearly **400** dog breeds. Most traits are rated on a scale from 1 to 5. The dataset includes details such as:  

- **Breed Group:** Classification based on temperament, skills, and history  
- **Height:** Average height (cm)  
- **Weight:** Average weight (kg)  
- **Life Span:** Typical lifespan range  
- **Adaptability:** Suitability for novice owners, apartment living, and hot/cold weather  
- **Friendliness:** Affection toward family, kids, strangers, and other dogs  
- **Health & Grooming:** Shedding level, drooling tendencies, grooming requirements  
- **Trainability:** Intelligence, barking tendencies, energy level

---

## App Features
- **Filter by breed group**, height, and weight using dropdowns and sliders  
- **Visualize breed differences** in traits like friendliness and trainability 

---

## Evaluation Metrics  
To assess your model, DataQuest provides key metrics:
- **Principal Component Analysis (PCA)**
  - *Cumulative Explained Variance:* Measures how much of the dataset’s total variance is captured by the selected principal components. Used to assess dimensionality reduction effectiveness.
- **K-Means Clustering: Explained Variance**
  - *Within-Cluster Sum of Squares (WCSS):* Indicates compactness; lower WCSS means points are closer to their cluster centroids.
  - *Silhouette Score:* Ranges from -1 to 1; higher values mean better-defined clusters (points are closer to their own cluster than others).
- **Hierarchical Clustering: Explained Variance**
  - *Silhouette Score:* Ranges from -1 to 1; higher values mean better-defined clusters (points are closer to their own cluster than others).

These metrics are displayed after training to help you compare model effectiveness.

---

## Visualization  
Visual tools help bring the data to life:
- **Explained Variance Plots** – Show how much information is retained with each principal component.
<img src="images/ExplainedVariance1.png" height="300">
<img src="images/ExplainedVariance2.png" height="300">

- **K-Means Cluster Optimization** – Visualize WCSS and silhouette scores for different cluster numbers.
<img src="images/KMeansPlots.png" height="300">

- **Dendrogram** – Display the hierarchical merging of clusters.
<img src="images/Dendrogram.png" height="300">

- **PCA Projections** – 2D and biplot visualizations of reduced data.
<img src="images/PCAProjection.png" height="300">
<img src="images/PCAProjectionBiplot.png" height="300">

---

## References  

### Machine Learning  
- *Grokking Machine Learning* by Luis G. Serrano  

### Dogfinder Dataset  
- [Dogfinder Dataset](https://www.kaggle.com/datasets/yonkotoshiro/dogs-breeds) – via Kaggle

