import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
data = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# Split features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
regression = LinearRegression()
regression.fit(X_train, y_train)

# Cross-validation
mse = cross_val_score(
    regression,
    X_train,
    y_train,
    scoring='neg_mean_squared_error',
    cv=5
)

print("Mean MSE:", np.mean(mse))

# Predictions
reg_pred = regression.predict(X_test)
print(reg_pred[:10])  # first 10 predictions

# checking the model outputs

residuals = reg_pred - y_test

plt.figure()
plt.hist(residuals, bins=50, rwidth=0.8)
plt.xlabel("Residuals (Prediction Error)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()