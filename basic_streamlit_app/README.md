# :dog: Welcome to DogFinder! :poodle:
*An Interactive Streamlit App*

## Project Overview  
**DogFinder** is a [**Streamlit**](https://streamlit.io/) app designed to track information across a range of dog breeds. The project enables interactive visualization of characteristics such as size, behavior, and friendliness.

---

## Getting Started

### Install Dependencies  
Install & import the necessary libraries with the following commands:
```bash
pip install pandas streamlit
```
```python
import pandas as pd
import streamlit
```

### Download the data
- Download my_data.csv and place it in the `data` folder within your project directory.

### Run the App
Run the app directly from this repository using the following command:
 ```
 streamlit run main.py
 ```
---

## The DogFinder Dataset
The app uses a cleaned dataset from [Kaggle](https://www.kaggle.com/datasets/yonkotoshiro/dogs-breeds) containing information about almost **400** different dog breeds. Most traits are presented on a scale from 1 to 5. The dataset captures numerous features, including:
- **Dog Breed Group:** the breed's assigned group based on temperament, skills, and history
- **Height:** average height in centimeters (cm)
- **Weight:** average weight in kilograms (kg)
- **Life Span**
- **Adaptability**: suitablility for novice owners, apartment living, and hot/cold weather
- **Friendliness**: affection toward family, kids, strangers, and other dogs
- **Health and Grooming**: amount of shedding, drool, required grooming
- **Trainability**: intelligence, barking tendencies, energy level
  
---

## App Features
- Use the dropdown and sliders to filter by dog breed group and maximum heights and weights
- Graph differences in traits such as friendliness and trainability.


