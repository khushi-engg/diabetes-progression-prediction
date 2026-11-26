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

# ==========================================
# EDA SUMMARY
# ==========================================

st.subheader("EDA Summary")

if st.button("Show EDA Summary"):

    st.write("=" * 50)

    st.write("Number of patient records:", df.shape[0])
    st.write("Number of clinical features:", len(features))
    st.write("Target variable: Diabetes progression")

    st.write("Missing values:", df.isnull().sum().sum())
    st.write("Duplicate records:", df.duplicated().sum())

   # ==========================================
# FEATURE CARDS
# ==========================================

st.write("### Clinical Features")

feature_columns = st.columns(5)

for i, feature in enumerate(features):
    with feature_columns[i % 5]:
        with st.container(border=True):
            st.markdown(f"**{feature.upper()}**")
            st.caption(f"Feature {i + 1}")

    st.write("### Target Statistics")

    st.dataframe(
        df["target"].describe().to_frame(),
        use_container_width=True
    )
