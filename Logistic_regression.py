import pandas as pd
import seaborn as sb
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = sb.load_dataset("iris")

print(df.head())

# Encode species
df["species"] = df["species"].map({
    "setosa":0,
    "versicolor":1,
    "virginica":2
})

print(df)

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Logistic Regression (multiclass)
model = LogisticRegression(solver="lbfgs", multi_class="auto")

parameters = {
    "C":[0.1,1,2,5,10],
    "max_iter":[100,200,300]
}

grid = GridSearchCV(model, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Predictions
pred = best_model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Save Model
pickle.dump(best_model, open("iris_model.pkl", "wb"))
print(df)
print("Model Saved Successfully")