# :dog: Welcome to DogFinder! :poodle:
*A Streamlit App for Dog Data Analysis*

## Project Overview  
**DogFinder** is a [**Streamlit**](https://streamlit.io/) app designed to track information across a range of dog breeds. The project enables interactive visualization of characteristics such as size, behavior, and friendliness.

---

## Getting Started

### Run the App Locally

#### Clone & Navigate to the Repository
```bash
git clone https://github.com/annamccartan3/McCartan-Data-Science-Portfolio.git
cd McCartan-Data-Science-Portfolio/basic_streamlit_app
```
#### Install Dependencies  
Install & import the necessary libraries with the following commands:
```bash
pip install streamlit pandas
```

#### Download the data
Download `my_data.csv` and place it in the `data` folder within your project directory.

#### Run the Streamlit App
```bash
 streamlit run main.py
```

---

## The DogFinder Dataset
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


