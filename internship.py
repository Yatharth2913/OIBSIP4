import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Display first rows
print("First 5 Rows:")
print(df.head())

# Features and labels
X = df['message']
y = df['label']

# Convert text into numerical vectors
cv = CountVectorizer()

X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

# -----------------------------
# Graph: Confusion Matrix
# -----------------------------
plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------
# Sample Prediction
# -----------------------------
sample_message = ["Congratulations! You won a free lottery ticket"]

sample_data = cv.transform(sample_message)

prediction = model.predict(sample_data)

print("\nSample Message Prediction:")

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")