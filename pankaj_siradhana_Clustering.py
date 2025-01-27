import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Load cleaned data
customers = pd.read_csv(r"D:\Data Science\New folder\cleaned Customers.csv")
transactions = pd.read_csv(r"D:\Data Science\New folder\cleaned Transactions.csv")

# Feature Engineering: Aggregate customer transaction data
customer_spending = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count'
}).reset_index()
customer_spending.columns = ['CustomerID', 'TotalSpent', 'PurchaseCount']

# Merge customer data with spending features
customer_profiles = customers.merge(customer_spending, on='CustomerID', how='left')
customer_profiles.fillna(0, inplace=True)  # Fill missing values

# Normalize features for clustering
scaler = StandardScaler()
features = customer_profiles[['TotalSpent', 'PurchaseCount']]
scaled_features = scaler.fit_transform(features)

# Determine the best number of clusters (2 to 10) using DB Index
db_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    db_scores.append(davies_bouldin_score(scaled_features, labels))

# Select the best K (lowest DB index)
best_k = k_range[np.argmin(db_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
customer_profiles['Cluster'] = kmeans.fit_predict(scaled_features)

# Save results
customer_profiles.to_csv("D:/Data Science/New folder/Customer_Clusters.csv", index=False)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_profiles['TotalSpent'], y=customer_profiles['PurchaseCount'], hue=customer_profiles['Cluster'], palette='viridis')
plt.title(f'Customer Clusters (K={best_k})')
plt.xlabel('Total Spent')
plt.ylabel('Purchase Count')
plt.show()

# Print final DB Index
print(f'Optimal Clusters: {best_k}, DB Index: {min(db_scores):.3f}')
