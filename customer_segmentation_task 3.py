
# Task 3: Clustering Analysis – Customer Segmentation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Basic inspection
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())
print("\nSummary Statistics:\n", df.describe())

# Data Preprocessing
X = df[['Age', 'Annual Income', 'Spending Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal k
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# Silhouette Scores
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f'Silhouette Score for k={k}: {score:.3f}')

# Apply KMeans with chosen k (assumed 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 2D Visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segmentation (2D PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Pairplot
sns.pairplot(df[['Age', 'Annual Income', 'Spending Score', 'Cluster']], hue='Cluster', palette='Set2')
plt.suptitle("Pair Plot of Features by Cluster", y=1.02)
plt.show()

# Display Cluster Centroids
centroids = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)
centroid_df = pd.DataFrame(centroids_original, columns=['Age', 'Annual Income', 'Spending Score'])
print("\nCentroids of Clusters:\n", centroid_df)

# Basic Recommendations
print("\n--- Recommendations ---")
for i in range(optimal_k):
    segment = df[df['Cluster'] == i]
    print(f"\nCluster {i}:")
    print(f" - Number of Customers: {len(segment)}")
    print(f" - Average Age: {segment['Age'].mean():.2f}")
    print(f" - Average Income: ${segment['Annual Income'].mean():.2f}")
    print(f" - Avg Spending Score: {segment['Spending Score'].mean():.2f}")
