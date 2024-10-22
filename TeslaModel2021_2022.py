import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('TESLA.csv')


df_cleaned = df.drop_duplicates()




plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['Close'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()


df_cleaned.info()


plt.figure(figsize=(12, 8))
df_cleaned[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].boxplot()
plt.title('Boxplot for Outlier Detection')
plt.show()


print(df_cleaned.isnull().sum())  # No missing values


plt.figure(figsize=(10, 8))
correlation_matrix = df_cleaned[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
plt.title('Correlation Matrix', fontsize=16)
cax = plt.matshow(correlation_matrix, cmap='coolwarm')
plt.colorbar(cax)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()


final_features = ['Open', 'High', 'Low', 'Volume']
X = df_cleaned[final_features]
y = df_cleaned['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'AdaBoost Regressor': AdaBoostRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor(),
    'SVM Regressor': SVR()
}


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {'MSE': mse, 'R2': r2}
    return results


model_results = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
results_df = pd.DataFrame(model_results).T


print(results_df)


best_model = LinearRegression()
best_model.fit(X_train_scaled, y_train)


import joblib
joblib.dump(best_model, 'best_model.pkl')


def predict_new_data(new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = best_model.predict(new_data_scaled)
    return prediction


new_data = pd.DataFrame({'Open': [300], 'High': [310], 'Low': [295], 'Volume': [80000000]})
predicted_close = predict_new_data(new_data)
print(f"Predicted Close Price: {predicted_close[0]}")




