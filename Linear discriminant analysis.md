# 🍷 Wine Classification Problem Using LDA

## 🧩 Problem Statement

We have a dataset containing **information about different wines**.  
For each wine, customers have been categorized into **three distinct market segments**:

- **Segment 1**
- **Segment 2**
- **Segment 3**

When a **new type of wine** appears, our goal is to **predict which customer segment** it will appeal to based on its characteristics.

---

## 🎯 Objective

> Use **Linear Discriminant Analysis (LDA)** to classify a new wine into one of the three predefined customer segments based on wine attributes.

---

## 📊 Dataset Example

| Alcohol | Malic Acid | Ash | Alcalinity | Magnesium | Segment |
|---------|------------|-----|------------|-----------|---------|
| 14.23   | 1.71       | 2.43| 15.6       | 127       | 1       |
| 13.20   | 1.78       | 2.14| 11.2       | 100       | 1       |
| 12.37   | 1.17       | 1.92| 19.6       | 78        | 2       |
| 12.04   | 4.30       | 2.38| 22.0       | 80        | 2       |
| 14.13   | 4.10       | 2.74| 24.5       | 96        | 3       |

---

## ❓ Why Linear Discriminant Analysis?

- **LDA** is a supervised classification technique.
- It works well when **predictor variables are continuous** and the classes are **known in advance**.
- It projects the data into a lower-dimensional space to **maximize class separability**.
- Especially useful when there are **more than two classes** (multi-class classification).

---

## 🧪 Model Training Process

1. **Import Libraries & Dataset**
2. **Split the Dataset** into training and test sets
3. **Scale the Features** for optimal performance
4. **Fit LDA** to extract the most discriminative features
5. **Train a Classifier** (e.g., Logistic Regression) on the transformed features
6. **Predict** the customer segment for a new wine

---

## 🤖 Example Code

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load dataset
df = pd.read_csv('wine.csv')
X = df.drop('Segment', axis=1).values
y = df['Segment'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Train classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict for a new wine
new_wine_features = [[13.5, 2.1, 2.4, 18.5, 110]]  # Example values
new_wine_scaled = sc.transform(new_wine_features)
new_wine_lda = lda.transform(new_wine_scaled)
predicted_segment = classifier.predict(new_wine_lda)

print(f"Predicted Segment: {predicted_segment[0]}")
```

---

## ✅ Outcome

- Predict **customer segment** for any new wine based on its attributes.
- Use LDA to ensure **maximum class separation** and better prediction accuracy.

---

## 🧠 Summary

By applying **Linear Discriminant Analysis**:
- We reduce dimensionality while preserving **class-discriminating information**.
- We improve classification accuracy for **multi-class problems** like this wine segmentation.
- The approach is interpretable and works well with structured tabular datasets.

