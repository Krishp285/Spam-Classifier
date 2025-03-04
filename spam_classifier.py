# Spam Classifier by [Your Name]
# Detects spam emails with machine learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("spambase.data", header=None)

# Show first 5 rows
print("First 5 rows of data:")
print(data.head())

# Count spam vs. not spam
print("\nSpam vs. Not Spam count:")
print(data[57].value_counts())

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check sizes
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Test a fake email
fake_email = np.zeros(57)
fake_email[0] = 0.5
fake_email[1] = 0.3
fake_email_scaled = scaler.transform([fake_email])
prediction = model.predict(fake_email_scaled)
print("\nFake email prediction:", "Spam" if prediction[0] == 1 else "Not Spam")