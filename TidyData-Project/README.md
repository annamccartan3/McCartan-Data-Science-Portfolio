# :medal_sports: Gold Standard  
*A Data Tidying and Visualization Project*  

## Project Overview  
**Gold Standard** is a Jupyter Notebook designed for cleaning and visualizing data from the **2008 Summer Olympics** in Beijing. The project follows [**tidy data**](https://vita.had.co.nz/papers/tidy-data.pdf) principles to structure data effectively, making it easier to analyze and visualize.  

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
pip install pandas seaborn matplotlib
```
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### Download the Data  
- Download olympics08.csv and place it in the `data` folder within your project directory.  
- Download the TidyData.ipynb Notebook file.  

### Run the Notebook  
- Open the Jupyter Notebook in your preferred environment. Options include:  
  - Jupyter Notebook/Jupyter Lab  
  - Google Colab  
  - VSCode (with Jupyter extension)  

---

## The Olympics Dataset  
The dataset used in this project is adapted from [this source](https://edjnet.github.io/OlympicsGoNUTS/2008/), which contains **1,875** records of Olympic medalists from the **2008 Summer Olympics**, covering **75 different events** across multiple sports.  

---

## Features & Functionality  

**Transform Untidy Data**  
- Import the raw dataset and export the cleaned version as **olympics08_cleaned.csv**   

**Aggregation & Analysis**  
- Group and analyze data by **event, gender, and medal type**

**Data Visualization**  
- Generate intuitive charts to explore medal distributions 

### Example Output  
**Bar Graph:** This graph shows the number of medals awarded for each event type.
![Image](https://github.com/user-attachments/assets/fbff3223-bae7-4cc8-a051-7a1cca853c03)
**Heatmap:** This heatmap displays the distribution of medals by event and gender.
![Image](https://github.com/user-attachments/assets/ecf345a5-85c3-4f7d-a28b-153b804365ed)

---

## References
- [**Tidy Data**](https://vita.had.co.nz/papers/tidy-data.pdf) by Hadley Wickham
- [**Pandas Cheat Sheet**](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
