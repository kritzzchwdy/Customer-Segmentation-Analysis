# ============================================================
# Project 2: Customer Segmentation Analysis
# Author: [Your Name]
# Description: Analyze customer data using K-Means clustering
#              to segment customers by income & spending behavior
# ============================================================

# ── STEP 1: Import Required Libraries ──────────────────────
import pandas as pd               # For data manipulation
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For plotting graphs
import seaborn as sns             # For beautiful visualizations
from sklearn.cluster import KMeans           # K-Means algorithm
from sklearn.preprocessing import StandardScaler  # Normalize data

import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   Customer Segmentation Analysis - Project 2")
print("=" * 55)


# ── STEP 2: Load Dataset ───────────────────────────────────
# Dataset: Mall_Customers.csv from Kaggle
# Columns: CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)

df = pd.read_csv("Mall_Customers.csv")

print("\n[1] Dataset Loaded Successfully!")
print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ── STEP 3: Explore the Dataset ────────────────────────────
print("\n[2] First 5 rows of the dataset:")
print(df.head())

print("\n[3] Dataset Info (column types, null counts):")
print(df.info())

print("\n[4] Basic Statistics:")
print(df.describe())

print("\n[5] Missing Values per Column:")
print(df.isnull().sum())


# ── STEP 4: Clean & Preprocess Data ────────────────────────
# Rename columns for easier access
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income',
    'Spending Score (1-100)': 'Spending_Score',
    'Genre': 'Gender'
}, inplace=True)

# Check for duplicates
print(f"\n[6] Duplicate Rows: {df.duplicated().sum()}")

# Drop CustomerID (not useful for analysis)
df_clean = df.drop(columns=['CustomerID'])

print("\n[7] Cleaned Dataset (first 5 rows):")
print(df_clean.head())


# ── STEP 5: Exploratory Data Analysis (EDA) ────────────────
print("\n[8] Running Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Data - Exploratory Analysis', fontsize=16, fontweight='bold')

# Plot 1: Age Distribution
axes[0, 0].hist(df_clean['Age'], bins=20, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')

# Plot 2: Annual Income Distribution
axes[0, 1].hist(df_clean['Annual_Income'], bins=20, color='coral', edgecolor='white')
axes[0, 1].set_title('Annual Income Distribution')
axes[0, 1].set_xlabel('Annual Income (k$)')
axes[0, 1].set_ylabel('Count')

# Plot 3: Spending Score Distribution
axes[1, 0].hist(df_clean['Spending_Score'], bins=20, color='mediumseagreen', edgecolor='white')
axes[1, 0].set_title('Spending Score Distribution')
axes[1, 0].set_xlabel('Spending Score (1-100)')
axes[1, 0].set_ylabel('Count')

# Plot 4: Gender Count
gender_counts = df_clean['Gender'].value_counts()
axes[1, 1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
               colors=['#FF9999', '#66B2FF'], startangle=90)
axes[1, 1].set_title('Gender Distribution')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("    EDA chart saved as 'eda_analysis.png'")


# ── STEP 6: Select Features for Clustering ─────────────────
# We use Annual Income and Spending Score for 2D clustering
X = df_clean[['Annual_Income', 'Spending_Score']]

print("\n[9] Features selected for K-Means:")
print("    - Annual Income (k$)")
print("    - Spending Score (1-100)")


# ── STEP 7: Find Optimal K using Elbow Method ──────────────
print("\n[10] Running Elbow Method to find optimal number of clusters...")

inertia = []   # Sum of squared distances (WCSS)
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', color='purple', linewidth=2, markersize=8)
plt.title('Elbow Method — Finding Optimal K', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Elbow chart saved as 'elbow_method.png'")
print("    → Optimal K = 5 (elbow bend point)")


# ── STEP 8: Apply K-Means Clustering (K=5) ─────────────────
print("\n[11] Applying K-Means Clustering with K=5...")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X)

print(f"    Cluster Centers:\n{pd.DataFrame(kmeans.cluster_centers_, columns=['Annual_Income','Spending_Score'])}")
print(f"\n    Customers per Cluster:")
print(df_clean['Cluster'].value_counts().sort_index())


# ── STEP 9: Visualize Clusters ─────────────────────────────
print("\n[12] Visualizing Customer Segments...")

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
labels = [
    'Cluster 0: Low Income, Low Spending',
    'Cluster 1: High Income, Low Spending',
    'Cluster 2: Average Income, Average Spending',
    'Cluster 3: Low Income, High Spending',
    'Cluster 4: High Income, High Spending'
]

plt.figure(figsize=(10, 7))

for i in range(5):
    cluster_data = df_clean[df_clean['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual_Income'],
        cluster_data['Spending_Score'],
        c=colors[i], label=labels[i], s=80, alpha=0.8, edgecolors='white', linewidths=0.5
    )

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],
            c='black', marker='X', s=200, zorder=10, label='Cluster Centers')

plt.title('Customer Segmentation (K-Means, K=5)', fontsize=15, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(loc='upper left', fontsize=8.5)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('customer_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Cluster chart saved as 'customer_clusters.png'")


# ── STEP 10: Interpret Clusters (Business Insights) ────────
print("\n[13] Business Insights from Clusters:")
print("-" * 55)

segment_descriptions = {
    0: ("Low Income, Low Spending",    "Budget-conscious customers. Offer discounts & value deals."),
    1: ("High Income, Low Spending",   "Wealthy but careful spenders. Target with premium loyalty programs."),
    2: ("Average Income, Avg Spending","Middle-ground customers. Respond well to seasonal promotions."),
    3: ("Low Income, High Spending",   "Impulsive spenders. Risk of debt. Offer EMI/installment options."),
    4: ("High Income, High Spending",  "VIP customers! High priority — offer exclusive memberships."),
}

for cluster_id, (segment, action) in segment_descriptions.items():
    count = len(df_clean[df_clean['Cluster'] == cluster_id])
    print(f"\n  Cluster {cluster_id} → {segment}")
    print(f"  Customers: {count}")
    print(f"  Strategy : {action}")

print("\n" + "=" * 55)
print("   Analysis Complete! Check saved PNG files.")
print("=" * 55)


# ── STEP 11: Save Final Segmented Data ─────────────────────
df_clean.to_csv('segmented_customers.csv', index=False)
print("\n[14] Segmented dataset saved as 'segmented_customers.csv'")
