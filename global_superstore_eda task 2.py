
# Project 1: Exploratory Data Analysis (EDA) - Global Superstore

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

# STEP 1: Load the dataset
df = pd.read_csv('global_superstore.csv')
print(df.head())

# STEP 2: Understand the dataset
print(df.info())
print(df.isnull().sum())

# STEP 3: Handle missing values
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
df['Profit'] = df['Profit'].fillna(df['Profit'].median())
df.dropna(inplace=True)

# STEP 4: Remove duplicate rows
df.drop_duplicates(inplace=True)

# STEP 5: Deal with outliers (using IQR method)
def remove_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

df = remove_outliers('Sales')
df = remove_outliers('Profit')

# STEP 6: Basic statistics
print(df[['Sales', 'Profit']].describe())
print(df[['Sales', 'Profit']].corr())

# STEP 7: Visualizations

# Histogram - Sales
plt.figure(figsize=(6,4))
sns.histplot(df['Sales'], bins=30, kde=True, color='blue')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Count')
plt.show()

# Histogram - Profit
plt.figure(figsize=(6,4))
sns.histplot(df['Profit'], bins=30, kde=True, color='green')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Count')
plt.show()

# Boxplot - Sales
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Sales'], color='orange')
plt.title('Boxplot - Sales')
plt.show()

# Boxplot - Profit
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Profit'], color='red')
plt.title('Boxplot - Profit')
plt.show()

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[['Sales', 'Profit']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Sales by Region
plt.figure(figsize=(10,5))
sns.barplot(data=df, x='Region', y='Sales', estimator=np.sum, ci=None)
plt.title('Total Sales by Region')
plt.xticks(rotation=45)
plt.show()

# Profit by Product Category
plt.figure(figsize=(10,5))
sns.barplot(data=df, x='Product Category', y='Profit', estimator=np.sum, ci=None)
plt.title('Total Profit by Product Category')
plt.xticks(rotation=45)
plt.show()
