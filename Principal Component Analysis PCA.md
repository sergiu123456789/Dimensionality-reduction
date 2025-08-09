# 🍷 PCA Problem: Predicting Customer Segments for New Wines

## 🧩 Problem Statement

A dataset contains **information about different wines** (e.g., chemical composition, acidity, alcohol content, etc.) and the **customer segment** that each wine appeals to.

- **Customer segments** are labeled as:  
  - Segment 1  
  - Segment 2  
  - Segment 3  

When a **new type of wine** is introduced, we want to **predict** which **customer segment** it will belong to.

---

## 🎯 Objective

> Use **Principal Component Analysis (PCA)** to reduce the dimensionality of wine features while preserving important patterns, and then use a **classification model** to predict the segment for a new wine.

---

## 📊 Why PCA?

- Wine datasets often contain **many correlated variables** (e.g., multiple chemical measurements).
- **PCA**:
  - Reduces the number of features while retaining most of the variance.
  - Removes multicollinearity, making models more stable.
  - Improves visualization in **2D/3D** for understanding wine distribution across segments.

---

## 🧪 Dataset Example

| Alcohol | Malic Acid | Ash  | Alcalinity | Magnesium | Total Phenols | Flavanoids | Nonflav. Phenols | Proanth. | Color Intensity | Hue  | OD280/OD315 | Proline | Segment |
|---------|------------|------|------------|-----------|---------------|------------|------------------|----------|-----------------|------|-------------|---------|---------|
| 14.23   | 1.71        | 2.43 | 15.6       | 127       | 2.80          | 3.06       | 0.28             | 2.29     | 5.64            | 1.04 | 3.92        | 1065    | 1       |
| 13.20   | 1.78        | 2.14 | 11.2       | 100       | 2.65          | 2.76       | 0.26             | 1.28     | 4.38            | 1.05 | 3.40        | 1050    | 2       |
| ...     | ...         | ...  | ...        | ...       | ...           | ...        | ...              | ...      | ...             | ...  | ...         | ...     | ...     |

---

## 🤖 Approach

1. **Import and Explore Data**
   - Load wine dataset
   - Check missing values, distributions

2. **Feature Scaling**
   - Standardize features (mean = 0, std = 1) before PCA

3. **Apply PCA**
   - Choose the number of components to retain (e.g., enough to capture 95% variance)
   - Transform original features into principal components

4. **Train a Classifier**
   - Use the principal components as inputs to a model such as:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)

5. **Predict New Wine Segment**
   - Transform the new wine’s features using the same PCA
   - Predict its segment using the trained classifier

---

## 📂 Example Code

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load data
data = load_wine()
X, y = data.data, data.target  # Segments are encoded as 0, 1, 2

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict new wine's segment
new_wine_features = scaler.transform([[13.5, 2.1, 2.5, 15.0, 110, 2.8, 3.0, 0.3, 2.0, 5.5, 1.0, 3.9, 1050]])
new_wine_pca = pca.transform(new_wine_features)
segment_pred = clf.predict(new_wine_pca)
print(f"Predicted Segment: {segment_pred[0] + 1}")  # +1 to match segment labels
```

---

## ✅ Outcome

- **Dimensionality reduction** simplifies the dataset without losing critical variance.
- A **classifier** trained on PCA-transformed data predicts which customer segment a wine will likely appeal to.
- HR/marketing teams can **target promotions** for new wines more effectively.

---

## 🧠 Summary

Using **PCA + Classification**:
- Reduces noise and redundancy in high-dimensional wine data.
- Improves interpretability of patterns across customer segments.
- Supports **data-driven marketing and segmentation** decisions.

