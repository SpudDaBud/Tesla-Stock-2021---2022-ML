import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# Set page configuration
st.set_page_config(page_title="Tesla Stock Price Prediction", layout="wide")


# Step 1: Load and Inspect Dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("TESLA.csv")
        data.drop_duplicates(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


data = load_data()
if data is None:
    st.stop()


st.title("Tesla Stock Price Prediction")
st.subheader("Dataset Sample")
st.write(data.sample(5))


# Display all column names
st.subheader("Column Names")
st.write(data.columns.tolist())


# Step 2: Problem Statement
st.subheader("Problem Statement")
st.write(f"Data Shape: {data.shape}")
st.write("We aim to predict the 'Close' price of Tesla stock based on historical stock data.")


# Target column (Close)
target_column = 'Close'


# Step 3: Target Distribution
st.subheader(f"Target Variable Distribution ({target_column})")
fig, ax = plt.subplots()
sns.histplot(data[target_column], kde=True, ax=ax)
st.pyplot(fig)


# Step 4: Data Exploration
st.subheader("Data Exploration")
st.write(data.describe())


# Step 5: Visual EDA for continuous variables
st.subheader("Exploratory Data Analysis")
numeric_cols = data.select_dtypes(include=[np.number]).columns


for col in numeric_cols:
    fig, ax = plt.subplots()
    sns.histplot(data[col], kde=True, ax=ax)
    st.pyplot(fig)


# Create a new DataFrame to store custom metrics
measures = pd.DataFrame(data[target_column])


# Adding a new column 'ten' with a constant value of 10
measures['ten'] = 10


measures['rolling_mean'] = measures[target_column].rolling(10).mean()


fig, ax = plt.subplots()
measures['rolling_mean'].plot(ax=ax, label='Rolling Mean (10 periods)', color='orange')
ax.set_title("Rolling Mean of Tesla 'Close' Prices")
ax.legend()
st.pyplot(fig)


# Step 6-7: Outlier and Missing Value Analysis
st.subheader("Handling Missing Values and Outliers")


# Handle missing values
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
st.write("Missing values have been handled.")


# Outlier analysis
st.write("Outlier Analysis:")
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
st.write(f"Number of outliers in each numeric column:\n{outliers}")


# Step 8: Feature Selection using Correlation for continuous variables
st.subheader("Correlation Matrix")
corr = data[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)


# Step 9: Data Preparation for ML
X = data[['Open', 'High', 'Low', 'Volume']]
y = data[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 10-12: Train Multiple Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'SVM': SVR()
}


st.subheader("Model Training and Evaluation")
results = {}
for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
    except Exception as e:
        st.warning(f"Error training {name} model: {str(e)}")


# Step 13: Model Selection
if results:
    best_model = min(results, key=lambda x: results[x]['MSE'])
    st.write(f"Best Model: {best_model}")
    st.write(pd.DataFrame(results).transpose())
else:
    st.error("No models were successfully trained. Please check your data.")
    st.stop()


# Step 14: Deployment using Streamlit
st.subheader(f"Make Predictions using {best_model}")
model = models[best_model]


# Create input fields dynamically based on features
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter value for {col}", value=float(data[col].mean()))


if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.write(f"Predicted {target_column}: {prediction[0]}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please ensure all input fields are filled correctly.")




