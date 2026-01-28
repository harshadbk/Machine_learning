# import libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score # tis is for cross validation purpose

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
# tis is for Ridge reression
from sklearn.model_selection import GridSearchCV

# creatin a datasets

# Simple synthetic data
# y = 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)      # 100 samples, 1 feature
y = 3 * X + np.random.randn(100, 1) # add noise

df = pd.DataFrame({
    'X': X.flatten(),
    'y': y.flatten()
})

# train test split 

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ordinary Linear Regression (no regularization)
lr = LinearRegression()
lr.fit(X_train, y_train)

cv_scores = cross_val_score(
    lr,
    X,          # use full dataset
    y,
    cv=5,       # 5 folds
    scoring='r2'
)

print(np.mean(cv_scores))

y_pred_lr = lr.predict(X_test)

print("Linear Regression")
print("Weight:", lr.coef_.flatten())
print("Bias:", lr.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))




# ridge and lasso 



ridge = Ridge()

# Hyperparameter grid
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Grid Search (fit ONLY on training data)
grid = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring='r2'
)

grid.fit(X_train, y_train)

# Best model
best_ridge = grid.best_estimator_

# Predictions
y_pred_ridge = best_ridge.predict(X_test)

print("----- Ridge Regression (After GridSearchCV) -----")
print("Best alpha:", grid.best_params_)
print("Weight:", best_ridge.coef_[0])
print("Bias:", best_ridge.intercept_[0])
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score:", r2_score(y_test, y_pred_ridge))





# For Lasso




lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

print("----- Lasso Regression -----")
print("Weight:", lasso.coef_[0])
print("Bias:", lasso.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R2 Score:", r2_score(y_test, y_pred_lasso))
print()


# sort all data

sorted_idx = np.argsort(X_test.flatten())

X_test_sorted = X_test[sorted_idx]

y_lr_sorted = y_pred_lr[sorted_idx]
y_ridge_sorted = y_pred_ridge[sorted_idx]
y_lasso_sorted = y_pred_lasso[sorted_idx]


# ===============================
# 7. Visualization
# ===============================
plt.figure(figsize=(8,6))

plt.scatter(X, y, color='gray', label='Data')

plt.plot(X_test_sorted, y_lr_sorted, 'r-', label='Linear Regression')
plt.plot(X_test_sorted, y_ridge_sorted, 'b-', label='Ridge')
plt.plot(X_test_sorted, y_lasso_sorted, 'g-', label='Lasso')

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Ridge vs Lasso Regression")
plt.legend()
plt.show()


