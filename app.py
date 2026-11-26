import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
# Load the Diabetes dataset
diabetes = load_diabetes()
def section_gap(height=35):
    st.write("")
    st.markdown(f"<div style='height:{height}px'></div>", unsafe_allow_html=True)
# Convert the dataset into a DataFrame
df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)
# ==========================================
# CUSTOM BUTTON STYLING
# ==========================================

st.markdown("""
<style>

div.stButton > button {
    background-color: #315D72;
    color: #FFFFFF;
    border: 1px solid #315D72;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background-color: #27485B;
    color: #FFFFFF;
    border: 1px solid #27485B;
}

div.stButton > button:focus {
    box-shadow: none;
    outline: none;
}

</style>
""", unsafe_allow_html=True)
# Add target column
df["target"] = diabetes.target


# Website title
st.title("Diabetes Progression Prediction")

# Define features
features = diabetes.feature_names

# Dataset Explorer

st.subheader(" The Diabetes Progression dataset.")

# Button to display first 10 rows
# ==========================================
# DATASET EXPLORER
# ==========================================

# ==========================================
# DATASET EXPLORER
# ==========================================

if st.button("Show Dataset", key="dataset_preview_btn"):

    st.dataframe(
        df,                      # Full dataset: all 442 rows
        use_container_width=True,
        height=400,              # Compact initial view (~10 rows)
        hide_index=True
    )

    st.download_button(
        label="Download Complete Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="diabetes_dataset.csv",
        mime="text/csv",
        key="download_dataset_btn"
    )

section_gap(30)
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
section_gap(30)
st.subheader("EDA Summary")

if st.button("Show EDA Summary", key="eda_summary_btn"):

    # Dataset Overview
    st.markdown("### Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", len(features))
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    section_gap(20)

    # Clinical Features
    st.markdown("### Clinical Features")

    feature_columns = st.columns(5)

    for i, feature in enumerate(features):
        with feature_columns[i % 5]:
            with st.container(border=True):
                st.markdown(f"**{feature.upper()}**")
                st.caption(f"Feature {i + 1}")

    section_gap(20)

    # Target Statistics
    st.markdown("### Target Statistics")

    target_stats = df["target"].describe().to_frame()

    st.dataframe(
        target_stats,
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
section_gap(30)
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
section_gap(30)
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
# MODEL COMPARISON
# ==========================================

# ==========================================
# MODEL COMPARISON SECTION
# ==========================================

# ==========================================
# MODEL COMPARISON
# ==========================================
section_gap(30)
st.header("Model Comparison")

if st.button("Show Model Comparison", key="model_comparison_btn"):

    st.write("Compare the performance of four machine learning regression models.")

    X = df[features].copy()
    y = df["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": make_pipeline(
            StandardScaler(), LinearRegression()
        ),
        "KNN Regression": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=200, random_state=42
        )
    }

    results = []
    predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best model
    best_model = results_df.loc[results_df["RMSE"].idxmin()]
    best_model_name = best_model["Model"]
    best_predictions = predictions[best_model_name]

    # Save data for other visualizations
    st.session_state["results_df"] = results_df
    st.session_state["best_model_name"] = best_model_name
    st.session_state["best_predictions"] = best_predictions
    st.session_state["y_test"] = y_test

    # Mark results as available
    st.session_state["comparison_done"] = True


# Display results after button has been clicked
if st.session_state.get("comparison_done", False):

    results_df = st.session_state["results_df"]

    st.subheader("Model Performance Comparison")

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # Best model
    st.subheader("Best Performing Model")

    best_model = results_df.loc[results_df["RMSE"].idxmin()]

    st.success(f"Best Model: {best_model['Model']}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", round(best_model["MAE"], 4))
    col2.metric("MSE", round(best_model["MSE"], 4))
    col3.metric("RMSE", round(best_model["RMSE"], 4))
    col4.metric("R² Score", round(best_model["R2 Score"], 4))
# ==============================
# VISUALIZATIONS SECTION
# ==============================


section_gap(30)

# ==========================================
# VISUALIZATIONS SECTION
# ==========================================

st.header("Visualizations")


# ==========================================
# VISUALIZATION 1: ACTUAL VS PREDICTED
# ==========================================

# VISUALIZATION 1: ACTUAL VS PREDICTED

if st.button("1. Actual vs Predicted Plot", key="viz_btn_1"):

    if "best_predictions" not in st.session_state:
        st.warning("Please click Show Model Comparison first.")

    else:
        y_test = st.session_state["y_test"]
        best_predictions = st.session_state["best_predictions"]
        best_model_name = st.session_state["best_model_name"]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(y_test, best_predictions, alpha=0.7)

        min_value = min(np.min(y_test), np.min(best_predictions))
        max_value = max(np.max(y_test), np.max(best_predictions))

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="red",
            linestyle="--",
            label="Perfect Prediction"
        )

        ax.set_xlabel("Actual Diabetes Progression")
        ax.set_ylabel("Predicted Diabetes Progression")
        ax.set_title(f"Actual vs Predicted ({best_model_name})")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
# ==========================================
# VISUALIZATION 2: RESIDUAL DISTRIBUTION
# ==========================================

# VISUALIZATION 2: RESIDUAL DISTRIBUTION

if st.button("2. Residual Distribution Plot", key="viz_btn_2"):

    if "best_predictions" not in st.session_state:
        st.warning("Please click Show Model Comparison first.")

    else:
        y_test = st.session_state["y_test"]
        best_predictions = st.session_state["best_predictions"]
        best_model_name = st.session_state["best_model_name"]

        residuals = np.array(y_test) - np.array(best_predictions)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(residuals, bins=20, edgecolor="black")

        ax.axvline(
            x=0,
            color="red",
            linestyle="--",
            label="Zero Error"
        )

        ax.set_xlabel("Residuals (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residual Distribution ({best_model_name})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==========================================
# VISUALIZATION 3: CORRELATION HEATMAP
# ==========================================

if st.button("3. Feature Correlation Heatmap", key="viz_btn_3"):

    fig, ax = plt.subplots(figsize=(12, 9))

    correlation = df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Feature Correlation Heatmap")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# VISUALIZATION 4: CLINICAL FEATURE IMPORTANCE
# ==========================================

if st.button("4. Clinical Feature Importance", key="viz_btn_4"):

    X_feature = df[features]
    y_feature = df["target"]

    rf_feature_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf_feature_model.fit(X_feature, y_feature)

    importance = rf_feature_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title("Clinical Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Clinical Feature")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# VISUALIZATION 5: MODEL METRIC COMPARISON
# ==========================================

# VISUALIZATION 5: MODEL PERFORMANCE COMPARISON

if st.button("5. Model Performance Comparison Charts", key="viz_btn_5"):

    if "results_df" not in st.session_state:
        st.warning("Please click Show Model Comparison first.")

    else:
        results_df = st.session_state["results_df"]

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))

        # MAE
        axes[0].bar(results_df["Model"], results_df["MAE"])
        axes[0].set_title("Model Comparison - MAE")
        axes[0].set_ylabel("MAE (Lower is Better)")
        axes[0].tick_params(axis="x", rotation=45)

        # MSE
        axes[1].bar(results_df["Model"], results_df["MSE"])
        axes[1].set_title("Model Comparison - MSE")
        axes[1].set_ylabel("MSE (Lower is Better)")
        axes[1].tick_params(axis="x", rotation=45)

        # RMSE
        axes[2].bar(results_df["Model"], results_df["RMSE"])
        axes[2].set_title("Model Comparison - RMSE")
        axes[2].set_ylabel("RMSE (Lower is Better)")
        axes[2].tick_params(axis="x", rotation=45)

        # R2
        axes[3].bar(results_df["Model"], results_df["R2 Score"])
        axes[3].set_title("Model Comparison - R²")
        axes[3].set_ylabel("R² (Higher is Better)")
        axes[3].tick_params(axis="x", rotation=45)

        plt.suptitle("Performance Comparison of Regression Models", fontsize=16)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
