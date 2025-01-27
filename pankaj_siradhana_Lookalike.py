import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load cleaned data
customers = pd.read_csv(r"D:\Data Science\New folder\Customers.csv")
products = pd.read_csv(r"D:\Data Science\New folder\Products.csv")
transactions = pd.read_csv(r"D:\Data Science\New folder\Transactions.csv")

# Feature Engineering: Aggregate customer transaction data
customer_spending = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count'
}).reset_index()
customer_spending.columns = ['CustomerID', 'TotalSpent', 'PurchaseCount']

# Merge customer data with spending features
customer_profiles = customers.merge(customer_spending, on='CustomerID', how='left')
customer_profiles.fillna(0, inplace=True)  # Fill missing values

# Normalize features for similarity measurement
scaler = StandardScaler()
features = customer_profiles[['TotalSpent', 'PurchaseCount']]
scaled_features = scaler.fit_transform(features)

# Train KNN model
knn = NearestNeighbors(n_neighbors=4, metric='euclidean')  # 4 to include the customer itself
knn.fit(scaled_features)

# Generate Lookalike Recommendations
lookalike_results = {}
customer_ids = customer_profiles['CustomerID'].tolist()
distances, indices = knn.kneighbors(scaled_features)

for idx, customer_id in enumerate(customer_ids[:20]):  # First 20 customers (C0001 - C0020)
    similar_customers = [(customer_ids[i], round(distances[idx][j], 3)) for j, i in enumerate(indices[idx]) if i != idx]
    lookalike_results[customer_id] = similar_customers[:3]

# Save to CSV
lookalike_df = pd.DataFrame(list(lookalike_results.items()), columns=['cust_id', 'lookalikes'])
lookalike_df.to_csv("D:/Data Science/New folder/Lookalike.csv", index=False)

print("Lookalike Model Completed. Results saved in Lookalike.csv.")
