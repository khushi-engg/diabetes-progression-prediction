import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Define features
features = diabetes.feature_names

# Dataset Explorer

st.write(" The Diabetes Progression dataset.")

# Button to display first 10 rows
if st.button("Show First 10 Rows"):
    st.dataframe(df.head(10), use_container_width=True)


st.subheader("Feature vs Diabetes Progression Scatter Plots")

if st.button("Show Scatter Plots"):

    fig = plt.figure(figsize=(15, 12))

    for i, feature in enumerate(features, 1):
        ax = fig.add_subplot(4, 3, i)

        sns.scatterplot(
            data=df,
            x=feature,
            y="target",
            ax=ax
        )

        ax.set_title(f"{feature} vs Diabetes Progression")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

