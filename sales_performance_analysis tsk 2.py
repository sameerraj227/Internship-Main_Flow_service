
# Project 2: Sales Performance Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure plots show up in Jupyter Notebook
# %matplotlib inline

# STEP 1: Load and inspect the dataset
df = pd.read_csv('sales_data.csv')

# View the first few rows
print(df.head())

# Shape and data types
print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# STEP 2: Data Cleaning

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
df['Profit'] = df['Profit'].fillna(df['Profit'].mean())
df['Discount'] = df['Discount'].fillna(df['Discount'].mean())

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# STEP 3: Exploratory Data Analysis (EDA)

# Time Series Plot - Sales over time
df_sorted = df.sort_values('Date')
plt.figure(figsize=(10,5))
plt.plot(df_sorted['Date'], df_sorted['Sales'], color='blue')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter Plot - Profit vs Discount
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='Discount', y='Profit', hue='Category')
plt.title('Profit vs Discount')
plt.show()

# Bar Plot - Sales by Region
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Region', y='Sales', estimator=np.sum, ci=None)
plt.title('Total Sales by Region')
plt.xticks(rotation=45)
plt.show()

# Bar Plot - Sales by Category
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Category', y='Sales', estimator=np.sum, ci=None)
plt.title('Total Sales by Category')
plt.xticks(rotation=45)
plt.show()

# Optional: Pie Chart - Sales by Region
region_sales = df.groupby('Region')['Sales'].sum()
plt.figure(figsize=(6,6))
region_sales.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Sales Share by Region')
plt.ylabel('')
plt.show()

# STEP 4: Predictive Modeling - Linear Regression

# Prepare the data
X = df[['Profit', 'Discount']]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
