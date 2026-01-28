import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.array([
    [50, 1],
    [200, 5],
    [30, 0],
    [300, 7],
    [80, 1],
    [250, 6],
    [20, 0],
    [400, 10],
    [34,1],
    [95,3],
    [150,4],
    [250,6]
])

y = np.array([
    0, 1, 0, 1, 0, 1, 0, 1,1,0,1,1
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)


new_email = np.array([[200, 5]])

prediction = model.predict(new_email)
print(prediction)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")


# analysis of model predictions

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))