import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,classification_report

from sklearn.metrics import confusion_matrix

df = sb.load_dataset('iris')

print(df.head())

print(df['species'].unique())

print(df.isnull().sum().sum())

df = df[df['species']!='setosa']

print(df)

df['species'] = df['species'].map({'versicolor':0,'virginica':1})

print(df)

# split dataset into dependent and independent variable

X = df.iloc[:,:-1]
print(X)

y = df.iloc[:,-1]
print(y)

# dividing dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# creating regression

regression = LogisticRegression()

parameters = {'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,7,8,9,10,20,30,40,50],'max_iter':[100,200,300,400,500]}

CR = GridSearchCV(regression,param_grid=parameters,scoring='accuracy',cv=5)

CR.fit(X_train,y_train)

print(CR.best_params_)

print(CR.best_score_)

# predictions

y_pred = CR.predict(X_test)

print(y_pred)

# Accuracy Score

score = accuracy_score(y_pred,y_test)
print(score)

clsssR = classification_report(y_pred,y_test)
print(clsssR)


# Make new predictions

new_data = pd.DataFrame({
    'sepal_length': [6.1, 5.8],
    'sepal_width': [2.8, 2.7],
    'petal_length': [4.7, 5.1],
    'petal_width': [1.2, 1.9]
})

new_predictions = CR.predict(new_data)
print("New Predictions:", new_predictions)

species_map = {0: 'versicolor', 1: 'virginica'}
print([species_map[p] for p in new_predictions])

# visualization

result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

result['Correct'] = result['Actual'] == result['Predicted']

counts = result['Correct'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    counts,
    labels=['Correct', 'Incorrect'],
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Correct vs Incorrect Predictions")
# below is line graph

dot_df = pd.DataFrame({
    'Sample': range(len(y_test)),
    'Actual': y_test.values,
    'Predicted': y_pred
})

plt.figure(figsize=(10,5))

sb.lineplot(data=dot_df, x='Sample', y='Actual', label='Actual')
sb.lineplot(data=dot_df, x='Sample', y='Predicted', label='Predicted')

plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.title("Actual vs Predicted Values (Line Graph)")


# Dot Plot

plt.figure(figsize=(10,5))

sb.scatterplot(
    data=dot_df,
    x='Sample',
    y='Actual',
    label='Actual',
    s=80
)

sb.scatterplot(
    data=dot_df,
    x='Sample',
    y='Predicted',
    label='Predicted',
    s=80,
    marker='X'
)

plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.title("Dot Plot: Actual vs Predicted")
plt.show()