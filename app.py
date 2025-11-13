import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Model Deployment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 Machine Learning Model Hub</h1>', unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.title("Model Selection")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Choose a Section",
    ["Home", "EDA Dashboard", "Linear Regression", "K-Means Clustering", 
     "KNN", "Neural Network", "Random Forest", "XGBoost"]
)

# Load data function
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("DATASET/Cleaned_DATA.csv")
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        return None

# Load model function
def load_model(model_name):
    import pickle, os
    model_path = os.path.join("models", f"{model_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        return model_data  # full dict (model + features)
    else:
        st.error(f"Model file {model_name}.pkl not found!")
        return None


# Home Page
if model_choice == "Home":
    st.markdown('<h2 class="sub-header">Welcome to the ML Model Hub!</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Available Models")
        st.markdown("""
        - **Linear Regression**: Predict continuous values
        - **K-Means Clustering**: Unsupervised grouping
        - **KNN**: K-Nearest Neighbors classification
        - **Neural Network**: Deep learning predictions
        - **Random Forest**: Ensemble learning
        - **XGBoost**: Gradient boosting
        - **EDA Dashboard**: Explore and visualize data
        """)
    
    with col2:
        st.markdown("### 📈 Dataset Overview")
        data = load_data()
        if data is not None:
            st.write(f"**Rows:** {data.shape[0]}")
            st.write(f"**Columns:** {data.shape[1]}")
            st.write("**Preview:**")
            st.dataframe(data.head(), use_container_width=True)
    
    st.markdown("---")
    st.info("👈 Select a model from the sidebar to get started!")
    
# EDA Dashboard
# EDA Dashboard
elif model_choice == "EDA Dashboard":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)
    data = load_data()
    if data is not None:
        df = data.copy()
        st.markdown("### Select EDA Category")
        choice = st.radio("Choose EDA Type", ["Univariate", "Bivariate", "Multivariate", "Time-Series"])

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

        # --- UNIVARIATE ---
        if choice == "Univariate":
            st.subheader("Univariate Analysis")

            # Select numeric column for histogram
            if numeric_cols:
                col = st.selectbox("Select a numeric column to view distribution", numeric_cols)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for histogram.")

            # Select categorical column for countplot
            if cat_cols:
                col = st.selectbox("Select a categorical column for frequency plot", cat_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
                ax.set_title(f"Count of {col}")
                st.pyplot(fig)
            else:
                st.info("No categorical columns available for countplot.")

        # --- BIVARIATE ---
        elif choice == "Bivariate":
            st.subheader("Bivariate Analysis")

            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X-axis column", numeric_cols)
                y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for bivariate analysis.")

        # --- MULTIVARIATE ---
        elif choice == "Multivariate":
            st.subheader("Multivariate Analysis")

            if len(numeric_cols) >= 2:
                st.markdown("#### Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

                st.markdown("#### Pairplot (first 4 numeric features)")
                fig = sns.pairplot(df[numeric_cols[:4]])
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for multivariate plots.")

        # --- TIME SERIES ---
        elif choice == "Time-Series":
            st.subheader("Time-Series Analysis")

            # Detect a date or year-like column
            time_col = None
            for c in df.columns:
                if "date" in c.lower() or "year" in c.lower():
                    time_col = c
                    break

            if time_col:
                y_col = st.selectbox("Select numeric column for time trend", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(x=df[time_col], y=df[y_col], ax=ax)
                ax.set_title(f"{y_col} over {time_col}")
                st.pyplot(fig)
            else:
                st.warning("No date or year column found for time-series analysis.")




# Linear Regression
elif model_choice == "Linear Regression":
    st.markdown('<h2 class="sub-header">📈 Linear Regression Model</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        st.markdown("### Input Features")
        
        # Create input fields based on your dataset
        # Adjust these according to your actual features
        cols = st.columns(3)
        input_data = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for idx, col in enumerate(numeric_cols[:-1]):  # Exclude target variable
            with cols[idx % 3]:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(data[col].mean()),
                    step=float(data[col].std()) / 10
                )
        
        if st.button("Predict", key="lr_predict"):
            model_data = load_model("linear_regression")
            if model_data is not None:
                model = model_data["model"]
                feature_names = model_data["feature_names"]
                categorical_columns = model_data["categorical_columns"]
                numeric_columns = model_data["numeric_columns"]
            
                # Convert user input to DataFrame
                input_df = pd.DataFrame([input_data])
            
                # Handle categorical encoding (same as training)
                input_encoded = pd.get_dummies(input_df, drop_first=True)
            
                # Reindex to match training features
                input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
            
                # Predict
                prediction = model.predict(input_encoded)
                st.success(f"### Predicted Value: {float(prediction[0]):.2f}")


# K-Means Clustering
elif model_choice == "K-Means Clustering":
    st.markdown('<h2 class="sub-header">🎯 K-Means Clustering</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        st.markdown("### Select Features for Clustering")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect(
            "Choose features",
            numeric_cols,
            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        )
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        if st.button("Perform Clustering", key="kmeans_cluster"):
            from sklearn.cluster import KMeans
            
            if len(selected_features) >= 2:
                X = data[selected_features]
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                result_df = data.copy()
                result_df['Cluster'] = clusters
                
                st.markdown("### Clustering Results")
                st.dataframe(result_df.head(10), use_container_width=True)
                
                st.markdown("### Cluster Distribution")
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                st.bar_chart(cluster_counts)
            else:
                st.warning("Please select at least 2 features for clustering.")

# KNN
elif model_choice == "KNN":
    st.markdown('<h2 class="sub-header">🎯 K-Nearest Neighbors</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        st.markdown("### Input Features")
        
        cols = st.columns(3)
        input_data = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for idx, col in enumerate(numeric_cols[:-1]):
            with cols[idx % 3]:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(data[col].mean()),
                    step=float(data[col].std()) / 10,
                    key=f"knn_{col}"
                )
        
        if st.button("Classify", key="knn_predict"):
            model_data = load_model("knn")
            if model_data is not None:
                model = model_data["model"]
                feature_names = model_data["feature_names"]
                categorical_columns = model_data["categorical_columns"]
                numeric_columns = model_data["numeric_columns"]
            
                # Convert user input to DataFrame
                input_df = pd.DataFrame([input_data])
            
                # Handle categorical encoding (same as training)
                input_encoded = pd.get_dummies(input_df, drop_first=True)
            
                # Reindex to match training features
                input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
            
                # Predict
                prediction = model.predict(input_encoded)
                st.success(f"### Predicted Value: {float(prediction[0]):.2f}")


# Neural Network
elif model_choice == "Neural Network":
    st.markdown('<h2 class="sub-header">🧠 Neural Network</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        st.markdown("### Input Features")
        
        cols = st.columns(3)
        input_data = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for idx, col in enumerate(numeric_cols[:-1]):
            with cols[idx % 3]:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(data[col].mean()),
                    step=float(data[col].std()) / 10,
                    key=f"nn_{col}"
                )
        
        if st.button("Predict", key="nn_predict"):
            model_data = load_model("neural_network")
            if model_data is not None:
                model = model_data["model"]
                feature_names = model_data["feature_names"]

                # Convert input to DataFrame
                input_df = pd.DataFrame([input_data])

                # One-hot encode user input
                input_encoded = pd.get_dummies(input_df, drop_first=True)

                # Align with training features
                input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

                # Debug info
                st.write(f"Model expects {len(feature_names)} features, got {input_encoded.shape[1]}")

                # Predict
                prediction = model.predict(input_encoded)

                # Display result
                pred_value = float(np.squeeze(prediction))
                st.success(f"### Predicted Value: {pred_value:.2f}")




# Random Forest
elif model_choice == "Random Forest":
    st.markdown('<h2 class="sub-header">🌲 Random Forest</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        st.markdown("### Input Features")
        
        cols = st.columns(3)
        input_data = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for idx, col in enumerate(numeric_cols[:-1]):
            with cols[idx % 3]:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(data[col].mean()),
                    step=float(data[col].std()) / 10,
                    key=f"rf_{col}"
                )
        
        if st.button("Predict", key="rf_predict"):
            model_data = load_model("random_forest")
            if model_data is not None:
                model = model_data["model"]
                feature_names = model_data["feature_names"]
                categorical_columns = model_data["categorical_columns"]
                numeric_columns = model_data["numeric_columns"]
            
                # Convert user input to DataFrame
                input_df = pd.DataFrame([input_data])
            
                # Handle categorical encoding (same as training)
                input_encoded = pd.get_dummies(input_df, drop_first=True)
            
                # Reindex to match training features
                input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
            
                # Predict
                prediction = model.predict(input_encoded)
                st.success(f"### Predicted Value: {float(prediction[0]):.2f}")

                
                # Feature importance if available
                # --- Feature Importance Section ---
                if hasattr(model, "feature_importances_"):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    st.subheader("Feature Importance")
                    st.bar_chart(importance_df.set_index('Feature'))
                else:
                    st.info("This model does not support feature importance visualization.")


# XGBoost
elif model_choice == "XGBoost":
    st.markdown('<h2 class="sub-header">🚀 XGBoost</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        st.markdown("### Input Features")
        
        cols = st.columns(3)
        input_data = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for idx, col in enumerate(numeric_cols[:-1]):
            with cols[idx % 3]:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(data[col].mean()),
                    step=float(data[col].std()) / 10,
                    key=f"xgb_{col}"
                )
        
        if st.button("Predict", key="xgb_predict"):
            model_data = load_model("xgboost")

            if model_data is not None:
                model = model_data["model"]
                feature_names = model_data["feature_names"]
            
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df, drop_first=True)
            
                # Align columns with training features
                input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

                # FIX: Convert the DataFrame to a NumPy array to match the training data type.
                input_data_numpy = input_encoded.to_numpy()

                prediction = model.predict(input_data_numpy) # Use the NumPy array for prediction

                st.success(f"### Predicted Value: {float(prediction[0]):.2f}")



# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit | ML Model Hub</div>",
    unsafe_allow_html=True
)