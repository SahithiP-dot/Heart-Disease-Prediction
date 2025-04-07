import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/sahithipuppala/Desktop/heart_disease_dataset.csv")

# Display the first few rows
df.head()
# Check for missing values
print(df.isnull().sum())

# Basic statistics
print(df.describe())

# Check dataset shape
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize Heart Disease variable
sns.countplot(x=df['Heart Disease'])
plt.title("Heart Disease Presence (1) vs Absence (0)")
plt.show()

from sklearn.preprocessing import StandardScaler

print(df.dtypes)
df = pd.get_dummies(df, columns=['Gender','Smoking','Alcohol Intake','Family History','Exercise Induced Angina','Diabetes','Obesity','Chest Pain Type'], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)  # Now, all values should be numeric
y= df['Heart Disease']
from sklearn.model_selection import train_test_split

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Random Forest model1
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

from sklearn.svm import SVC

# Train SVM model2
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Predict on test data
y_pred_svm = svm_model.predict(X_test)

# Evaluate model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")

import numpy as np

# Get feature importances
feature_importances = rf_model.feature_importances_
features = df.drop(columns=['Heart Disease']).columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10,5))
print(len(feature_importances))
print(len(features))
print(sorted_indices)
sorted_indices = np.argsort(feature_importances)[::-1][:len(features)]
print("Length of feature_importances:", len(feature_importances))
print("Length of features:", len(features))
print("Max index in sorted_indices:", max(sorted_indices))  # Check if it's too high
print("Sorted Indices:", sorted_indices)
# First ensure we don't exceed the array bounds
n_features = len(features)
valid_indices = [i for i in sorted_indices if i < n_features]

# Now plot only the valid indices
sns.barplot(x=feature_importances[valid_indices],
            y=np.array(features)[valid_indices])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()