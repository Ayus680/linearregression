import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report

# Load the dataset
df = load_breast_cancer()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names

# Display the first few rows of the dataset
print(dataset.head())

# Data exploration
print("Dataset shape:", dataset.shape)
print("Dataset description:\n", dataset.describe())
print("Class distribution:\n", pd.Series(df.target).value_counts())

# Visualizing the data distribution
plt.figure(figsize=(10, 6))
sns.histplot(df.target, kde=False, bins=2)
plt.title("Distribution of Target Variable")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix
corr_matrix = dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Define the features and target variable
X = dataset
y = df.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Linear Regression model
regression = LinearRegression()

# Fit the model to the training set
regression.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(regression, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Make predictions on the test set
y_pred = regression.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted Values')
plt.plot(y_test, y_test, color='red', linewidth=2, label='Actual Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# Confusion Matrix
y_pred_class = [1 if pred > 0.5 else 0 for pred in y_pred]
cm = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred_class)
print("Classification Report:\n", report)

# Residuals plot
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_pred, lowess=True, color='green')
plt.xlabel('Actual')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual')
plt.show()

# Feature importance visualization
importances = np.abs(regression.coef_)
feature_names = dataset.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
