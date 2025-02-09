import streamlit as st
import pandas as pd

# app should display:
    # A title using st.title()
    # A short description of what the app does.
    # A sample DataFrame (to be loaded from a CSV, you may use something like Palmer's Penguins here Download Palmer's Penguins here).
    # Interactive filtering options (e.g., dropdowns, sliders, etc.).

# title using st.title()
st.title("DogFinder!")
st.subheader("A Streamlit app for tracking dog breeds.")
# short description of what the app does
st.write("Use the dropdown and sliders to filter by dog breed group and maximum heights and weights.")

# sample DataFrame of dog breed data, loaded from a CSV
dog_df = pd.read_csv("basic_streamlit_app/data/my_data.csv")
# drop unnecessary metrics
dog_df = dog_df.drop(["Detailed Description Link", "Height", "Weight", "Life Span"], axis=1)

# interactive filtering options
    
    # dropdown
group = st.selectbox("Select a Dog Breed Group", dog_df["Dog Breed Group"].unique())
    # sliders
height = st.slider("Choose a maximum height:",
                    min_value = dog_df["Avg. Height, cm"].min(),
                    max_value = dog_df["Avg. Height, cm"].max())
weight = st.slider("Choose a maximum weight:",
                    min_value = dog_df["Avg. Weight, kg"].min(),
                    max_value = dog_df["Avg. Weight, kg"].max())

# filter dataframe based on user choices
st.write(f"{group} under {height} cm and {weight} kg:")
st.dataframe(dog_df[(dog_df['Dog Breed Group']==group)&
                    (dog_df['Avg. Height, cm'] <= height)&
                    (dog_df['Avg. Weight, kg'] <= weight)])