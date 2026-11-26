import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()

# Convert the dataset into a DataFrame
df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

# Add target column
df["target"] = diabetes.target


# Website title
st.title("Diabetes Progression Prediction")



# Dataset Explorer

st.write(" The Diabetes Progression dataset.")

# Button to display first 10 rows
if st.button("Show First 10 Rows"):
    st.dataframe(df.head(10), use_container_width=True)
