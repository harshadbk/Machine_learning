from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    'StudyHours':[1,2,3,4,5],
    'TestScore':[40,50,60,70,80]
}

df = pd.DataFrame(data)

# standard scaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

print("Standard scaler output:")
print(pd.DataFrame(standard_scaled, columns=['StudyHours','TestScore']))


# min max scaler

minmaxscaler = MinMaxScaler()
minmaxscaled = minmaxscaler.fit_transform(df)

print("\nMinMax Scaled Output")

print(pd.DataFrame(minmaxscaled, columns=['StudyHours','TestScore']))

x = df[['StudyHours']]
y = df[['TestScore']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=48)

print("Training data")
print(x_train)

print("Testing data")
print(x_test)

print("Training data")
print(y_train)

print("Testing data")
print(y_test)