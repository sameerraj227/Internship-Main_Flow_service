
# Task 1: Predicting Student Pass/Fail using Logistic Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset creation
data = {
    'Study Hours': [5, 10, 3, 8, 15, 1, 7, 12, 0, 4, 11],
    'Attendance': [70, 85, 60, 80, 95, 50, 75, 90, 30, 65, 88],
    'Pass':       [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# 1. Data Exploration
print("\n--- Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe())

# Outlier check
sns.boxplot(data=df[['Study Hours', 'Attendance']])
plt.title("Boxplot to Check Outliers")
plt.show()

# Scatter plot to visualize trends
sns.scatterplot(x='Study Hours', y='Attendance', hue='Pass', data=df)
plt.title("Study Hours vs Attendance (Color = Pass/Fail)")
plt.show()

# 2. Model Training
X = df[['Study Hours', 'Attendance']]
y = df['Pass']

# Splitting into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Model Accuracy ---")
print(f"Accuracy: {accuracy:.2f}")

print("\n--- Confusion Matrix ---")
print(cm)

# 4. Insights
print("\n--- Insights ---")
print("Higher study hours and better attendance generally lead to passing.")
