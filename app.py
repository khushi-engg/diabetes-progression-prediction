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
# ==========================================
# CALCULATE FEATURE CORRELATION
# ==========================================

target_corr = df.corr()["target"].drop("target")

# Sort features by absolute correlation strength
target_corr = target_corr.reindex(
    target_corr.abs().sort_values(ascending=False).index
)
# ==========================================
# CORRELATION RANKING
# ==========================================

st.subheader("Correlation of Clinical Features with Diabetes Progression")

if st.button("Show Correlation Ranking"):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=target_corr.values,
        y=target_corr.index,
        palette="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation of Clinical Features with Diabetes Progression")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Clinical Features")
    ax.axvline(x=0, color="black", linestyle="--")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
# Select the top 5 features based on correlation strength
top_features = target_corr.abs().sort_values(ascending=False).head(5).index
# ==========================================
# TOP 5 CLINICAL FEATURES
# ==========================================

st.subheader("Top 5 Clinical Features vs Diabetes Progression")

if st.button("Show Top 5 Feature Plots"):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):

        sns.scatterplot(
            data=df,
            x=feature,
            y="target",
            alpha=0.6,
            ax=axes[i]
        )

        axes[i].set_title(f"{feature} vs Progression")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Diabetes Progression")

    # Remove the unused 6th subplot
    fig.delaxes(axes[5])

    fig.suptitle("Top 5 Clinical Features vs Diabetes Progression")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Strongest related feature
    st.markdown("### Strongest Related Feature")

    st.success(f"**Feature:** {target_corr.index[0].upper()}")

    st.metric(
        "Correlation Value",
        round(target_corr.iloc[0], 3)
    )

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# MODEL COMPARISON - FOUR REGRESSION MODELS
# ==========================================

st.header("Model Comparison")

st.write("Compare the performance of four machine learning regression models.")

# Prepare features and target
X = df[features].copy()
y = df["target"].copy()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize four models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "KNN Regression": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    ),

    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest Regression": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

if st.button("Train & Compare Models"):

    results = []
    predictions = {}

    for name, model in models.items():

        # Train model
        model.fit(X_train, y_train)

        # Predict test data
        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Save results
        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Display model comparison
    st.subheader("Model Performance Comparison")

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True
    )

