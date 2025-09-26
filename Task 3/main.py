from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt 
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve


# Generating and labelling synthetic data
X_real, _y = make_blobs(n_samples = 200, centers = 1, 
                        cluster_std = 2, n_features = 2, random_state = 42)
y_real = np.ones(X_real.shape[0])


X_fake = np.random.uniform(low=-6, high=6, size=(200, 2))
y_fake = np.zeros(X_fake.shape[0])

X = np.vstack((X_real, X_fake))
y = np.hstack((y_real, y_fake))

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Initaite Model
logreg = LogisticRegression()

# Fit Model
logreg.fit(X_train, y_train)

# Predict
y_pred = logreg.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", cnf_matrix)

# ===== ROC & AUC =====
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC score:", roc_auc)

# ===== Plot ROC Curve =====
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="gray", linestyle="--")  # random chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()


# ========== Plot class points ==========
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)

# ========== Plot decision boundary ==========
# Create a grid of points
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict class for each point in grid
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.title("Logistic Regression Class Separation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()



