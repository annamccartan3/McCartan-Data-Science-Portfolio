import streamlit as st
import pandas as pd

# app should display:
    # A title using st.title()
    # A short description of what the app does.
    # A sample DataFrame (to be loaded from a CSV, you may use something like Palmer's Penguins here Download Palmer's Penguins here).
    # Interactive filtering options (e.g., dropdowns, sliders, etc.).

# title using st.title()
st.title("DogFinder")

# short description of what the app does
st.write("A Streamlit app for tracking dog breeds in the United States of America.")

# sample DataFrame of dog breed data, loaded from a CSV
dog_df = pd.read_csv("data/sample_data.csv")
st.dataframe(dog_df)

# interactive filtering options
    # dropdowns

    # sliders