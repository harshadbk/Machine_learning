from sklearn.svm import SVR
import numpy as np

# Training data (features)
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

# Target values
y = np.array([2, 3, 4, 5, 6])

model = SVR(kernel='rbf') # Radial Basis Function

model.fit(X,y)

prediction = model.predict([[4,4]]).round(2)
print(prediction)