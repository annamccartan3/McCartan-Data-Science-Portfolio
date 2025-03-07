# :medal_sports: Gold Standard  
*A Data Tidying and Visualization Project*  

## Project Overview  
**Gold Standard** is a Jupyter Notebook designed for cleaning and visualizing data from the **2008 Summer Olympics** in Beijing. The project follows [**tidy data principles**](https://vita.had.co.nz/papers/tidy-data.pdf) to structure data effectively, making it easier to analyze and visualize.  

### What is Tidy Data?  
Tidy data principles, as outlined by **Hadley Wickham**, emphasize organizing data so that:  
- Each variable has its own column  
- Each observation has its own row  
- Each type of observational unit is stored in a separate table  

By applying these principles, this project transforms messy Olympic data into a structured format, enabling more efficient analysis.  

---

## Getting Started

### Install Dependencies  
Install & import the necessary libraries with the following commands:
```bash
pip install pandas numpy matplotlib  
```
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
```

### Download the Data  
- Download [**olympics08.csv**](TidyData-Project/data/olympics08.csv) and place it in the `data` folder within your project directory.  
- Download the [**TidyData.ipynb**](TidyData-Project/TidyData.ipynb) Notebook file.  

### Run the Notebook  
- Open the Jupyter Notebook in your preferred environment. Options include:  
  - Jupyter Notebook/Jupyter Lab  
  - Google Colab  
  - VSCode (with Jupyter extension)  

---

## The Olympics Dataset  
The dataset used in this project is adapted from [this source](https://edjnet.github.io/OlympicsGoNUTS/2008/). It contains **1,875** records of Olympic medalists from the **2008 Summer Olympics**, covering **75 different events** across multiple sports.  

---

## Features & Functionality  

**Transform Untidy Data**  
- Import the raw dataset and export the cleaned version as **olympics08_cleaned.csv**   

**Aggregation & Analysis**  
- Group and analyze data by **event, gender, and medal type**

**Data Visualization**  
- Generate interactive charts to explore medal distributions 

### Example Output  
*(Add screenshots of visualizations or a cleaned dataset preview for better presentation)*  

---

## References
- **Tidy Data Paper** by Hadley Wickham: [Read Here](https://vita.had.co.nz/papers/tidy-data.pdf)  
- **Pandas Cheat Sheet**: [View Here](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
