# 💧 Water Reservoir Analysis & Efficiency Predictor

## 🌍 Project Overview
This project focuses on predicting and understanding **how efficiently water reservoirs store and manage water** using Machine Learning and Deep Learning models.  
It studies how features like rainfall, inflow, evaporation and reservoir level affect the performance of Indian reservoirs.

Efficient water management is crucial for agriculture, industries, urban planning and drought risk mitigation.  
This project supports better **planning, decision support and water resource optimization**.

---

## 📊 Dataset Details

| Property         | Value |
|------------------|-------|
| Source           | Central Water Commission (Govt. of India) |
| File Used        | `Cleaned_DATA.csv` |
| Size             | ~53,000 rows × 10 columns |

### Dataset Columns:
- Country, State, District  
- Year, Month  
- Reservoir Basin Name, Reservoir Name  
- Full Reservoir Capacity (BCM)  
- Reservoir Water Level (M)  
- Reservoir Water Storage (BCM)

### Preprocessing Steps:
- Removed duplicates and missing values  
- Standardized column names  
- One-hot encoded categorical features  
- Scaled numeric features where required (KNN, NN, K-Means)

---

## ⚙️ Algorithms & Methods

| Model | Type | Purpose |
|--------|------|----------|
| Linear Regression | Baseline | Understand linear behavior |
| Random Forest | Ensemble | Handles nonlinearities & ranks feature importance |
| XGBoost | Gradient Boosting | Performance + regularization |
| KNN | Distance Based | Prediction based on nearest neighbors |
| Neural Network | Deep Learning | Captures complex nonlinear patterns |
| K-Means Clustering | Unsupervised | Reveals reservoir group behavior |

**Libraries used:**  
`pandas`, `numpy`, `sklearn`, `xgboost`, `tensorflow`, `matplotlib`, `seaborn`, `streamlit`

---

## 💻 Project Structure

| Notebook | Purpose |
|----------|----------|
| DataScience_EDA.ipynb | Complete exploratory analysis |
| LinearRegression.ipynb | Baseline model |
| RandomForest.ipynb | Best accuracy model |
| XGBooster.ipynb | Boosting optimization |
| KNN.ipynb | Distance-based modeling |
| NeuralNetwork.ipynb | Deep learning model experiments |
| K_MeansClustering.ipynb | Grouping reservoirs for insights |

---

## 🔬 Experimental Results

| Model | MAE | MSE | RMSE | R² Score | Remarks |
|-------|-----|-----|------|----------|---------|
| Linear Regression | 0.056 | 0.006 | 0.078 | 0.94 | Good baseline |
| Random Forest     | 0.005 | 0.003 | 0.057 | 0.99 | Best performing |
| XGBoost           | 0.053 | 0.027 | 0.164 | 0.93 | Stable & robust |
| KNN               | 0.039 | 0.029 | 0.173 | 0.93 | Works well with scaling |
| Neural Network    | ~0.11 | — | — | — | Needs more tuning |
| K-Means           | — | — | — | — | Formed 4 meaningful clusters |

### Key Insights
- **Random Forest & XGBoost** → Top performers  
- Most important feature: **Reservoir Water Level (M)**  
- Capacity & Rainfall-based features strongly influence efficiency  
- Neural Network needs deeper tuning (epochs / layers)

---

## 🧩 Project Architecture

```mermaid
flowchart LR
A[Raw Dataset] --> B[Data Cleaning & Preprocessing]
B --> C[EDA & Visualization]
C --> D[Model Training - ML Notebooks]
D --> E[Saved Models]
E --> F[Streamlit App]
F --> G[User Interaction & Predictions]
```

---

## 📚 References
- Central Water Commission, Government of India  
- scikit-learn Documentation  
- TensorFlow / Keras Documentation  
- XGBoost Documentation  

---

## 👨‍💻 Author
**Gautam Sukhani**  
Final Year Data Science Student  
**Skills:** Python | Machine Learning | XGBoost | TensorFlow

---

## 🌟 Highlights
- End-to-End Applied ML + Web Deployment  
- Multiple ML + DL Models implemented  
- Complete Exploratory Data Analysis (EDA)  
- Streamlit Web UI for Live Predictions
