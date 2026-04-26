# 🛍️ Customer Segmentation Analysis
> **Project 2 | Data Analytics**  
> Segmenting customers using K-Means Clustering based on Income & Spending Behavior

---

## 📌 Objective

Analyze customer data and divide users into meaningful groups based on their **Annual Income** and **Spending Score**. This helps businesses understand which type of customers to target with specific marketing strategies.

---

## 📂 Project Structure

```
customer-segmentation/
│
├── customer_segmentation.py       ← Main Python script
├── customer_segmentation.ipynb    ← Jupyter Notebook version
├── Mall_Customers.csv             ← Dataset (from Kaggle)
├── requirements.txt               ← Required Python libraries
│
├── outputs/
│   ├── eda_analysis.png           ← Exploratory Data Analysis charts
│   ├── elbow_method.png           ← Elbow method graph (K selection)
│   ├── customer_clusters.png      ← Final cluster scatter plot
│   └── segmented_customers.csv   ← Final dataset with cluster labels
│
└── README.md                      ← Project documentation
```

---

## 📊 Dataset

- **Source:** [Kaggle — Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Columns:**
  | Column | Description |
  |---|---|
  | CustomerID | Unique customer identifier |
  | Gender | Male / Female |
  | Age | Age of the customer |
  | Annual Income (k$) | Annual income in thousand dollars |
  | Spending Score (1-100) | Score assigned by the mall based on spending behavior |

---

## 🔧 Steps Performed

1. **Load Dataset** — Read CSV using Pandas
2. **Clean & Preprocess** — Rename columns, check nulls/duplicates, drop ID column
3. **EDA (Exploratory Data Analysis)** — Histograms and pie chart to understand distributions
4. **Elbow Method** — Determine optimal number of clusters (K=5)
5. **K-Means Clustering** — Fit model and assign cluster labels
6. **Visualize Clusters** — Scatter plot of Income vs Spending Score by cluster
7. **Interpret Clusters** — Generate actionable business insights per segment
8. **Export Results** — Save segmented data to CSV

---

## 📈 Cluster Results & Business Insights

| Cluster | Segment | Strategy |
|---|---|---|
| 0 | Low Income, Low Spending | Offer discounts & value deals |
| 1 | High Income, Low Spending | Target with premium loyalty programs |
| 2 | Average Income, Average Spending | Seasonal promotions |
| 3 | Low Income, High Spending | EMI / installment options |
| 4 | High Income, High Spending ⭐ | VIP — Exclusive membership offers |

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.x | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Basic plotting |
| Seaborn | Enhanced visualizations |
| Scikit-learn | K-Means clustering algorithm |

---

## 🚀 How to Run

**Step 1: Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Run the Python script**
```bash
python customer_segmentation.py
```

**OR open the Jupyter Notebook:**
```bash
jupyter notebook customer_segmentation.ipynb
---

## 👩‍💻 Author

**Kritika Chaudhary**  
B.Tech CSE — Semester IV  
Data Analytics Project

---

## 📄 License

This project is for educational purposes only.
