# 🍷 Wine Customer-Segment Prediction using Kernel PCA

## Problem Statement
Information (features) about wines is provided. Each wine is labeled with the customer segment that prefers it (segment **1**, **2**, or **3**).  
When a **new wine** appears, predict **which customer segment** it will belong to using **Kernel PCA** as a feature-extraction / dimensionality-reduction step before classification.

---

## Objective
- Use **Kernel Principal Component Analysis (Kernel PCA)** to capture non-linear structure in wine feature space.
- Feed the Kernel PCA transformed features into a classifier to predict the customer segment (1, 2, or 3).
- Evaluate model performance with standard classification metrics.

---

## Data (example)
Typical columns might include:
- Physico-chemical / sensory features (e.g., acidity, sugar, pH, alcohol, aroma scores, color intensity, etc.)
- `segment` — target label with values `{1, 2, 3}`

Example small schema:
```
| fixed_acidity | volatile_acidity | citric_acid | residual_sugar | pH | alcohol | ... | segment |
```

---

## Approach / Pipeline
1. **Load data** (CSV, database, etc.).  
2. **Exploratory Data Analysis (EDA)**: check distributions, missing values, correlations, class balance.  
3. **Preprocessing**:
   - Handle missing values (impute or drop).
   - Encode categorical features (if any).
   - Feature scaling (Kernel PCA and many classifiers require scaled features).
4. **Kernel PCA**:
   - Choose kernel (`rbf`, `poly`, `sigmoid`, or `cosine`).
   - Select `gamma` (for RBF) or `degree` (for poly) and number of components.
   - Transform features to new low-dimensional non-linear subspace.
5. **Classifier**:
   - Train a classifier on Kernel PCA outputs (e.g., SVM, Random Forest, Logistic Regression, or Gradient Boosting).
6. **Evaluation**:
   - Train/test split or cross-validation.
   - Metrics: accuracy, precision, recall, F1 (macro or weighted), confusion matrix.
   - If needed, use stratified splitting because of class imbalance.
7. **Prediction**:
   - For a new wine, run the same preprocessing → Kernel PCA transform → classifier.predict → return predicted segment.
8. **Explainability**:
   - Visualize Kernel PCA components (2D/3D) colored by segment.
   - Use class-probabilities or calibration to give confidence in predictions.

---

## Recommended Metrics
- **Accuracy**
- **Precision / Recall / F1-score** (macro and weighted)
- **Confusion matrix**
- (Optional) ROC-AUC per class (one-vs-rest)

---

## Suggested Hyperparameters to Tune
- Kernel PCA:
  - `kernel`: `rbf`, `poly`, `sigmoid`, `cosine`
  - `n_components`: e.g., 2, 5, 10, 20
  - `gamma` (for `rbf`/`poly`), `degree` (for `poly`)
- Classifier (example SVM):
  - `C`, `kernel` (linear / rbf), `gamma`
- Use GridSearchCV or RandomizedSearchCV with cross-validation.

---

## Example Python (scikit-learn) — Full Pipeline

```python
# Requirements: scikit-learn, pandas, numpy, matplotlib (optional)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv("wines.csv")   # replace with your file
X = df.drop(columns=["segment"]).values
y = df["segment"].values

# 2. Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Evaluate
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 4. Predict on a new wine example
new_wine = np.array([[7.1, 0.3, 0.2, 2.6, 5.1, 11.4]])  # replace with your features
pred_segment = best_model.predict(new_wine)
pred_proba = best_model.predict_proba(new_wine) if hasattr(best_model, "predict_proba") else None
print("Predicted segment:", pred_segment, "Probabilities:", pred_proba)
```

---

## Practical Notes & Tips
- **Scaling is essential** before Kernel PCA (and usually before SVM).
- Kernel PCA can be **computationally expensive** for very large datasets (it requires kernel matrix computations O(n²)). Consider subsampling or approximate kernel methods for large n.
- Kernel PCA does **not** directly produce inverse transforms for all kernels; if you need reconstructed features, set `fit_inverse_transform=True` and use a compatible kernel where possible.
- If interpretability is important, combine KPCA visualization with simple, interpretable classifiers (e.g., logistic regression) or use model-agnostic explainability tools.
- If class imbalance exists, prefer stratified CV and consider class weighting or resampling.

---

## Outcome
A working system that:
- Maps wine feature vectors to a low-dimensional non-linear feature space via Kernel PCA.
- Classifies wines into customer segments 1/2/3 with improved separation for non-linear relationships.
- Allows HR/marketing/product teams to predict which customer segment a new wine will target and support strategy decisions.
