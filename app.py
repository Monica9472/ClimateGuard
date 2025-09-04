# ClimateGuard - Wildfire Prediction
# Author: Monica Yuol Manyok

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the wildfire dataset
df = pd.read_csv("California_Fire_Incidents.csv")

print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Features (X) and target (y)
X = df[["AcresBurned", "Latitude", "Longitude", "Fatalities", "Injuries"]].fillna(0)
y = df["Status"].fillna("Unknown")

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Try a sample prediction
sample = [[50000, 37.5, -120.0, 2, 10]]  # 50,000 acres, central CA, 2 fatalities, 10 injuries
prediction = model.predict(sample)
print(f"Prediction for sample wildfire: Status = {prediction[0]}")

# ----------------------------
# 📊 Visualizations
# ----------------------------

# Visualization 1: Wildfire Status Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Status", data=df, palette="Reds")
plt.title("Wildfire Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Visualization 2: Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features, palette="rocket")
plt.title("Feature Importance in Wildfire Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
