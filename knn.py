from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = [[170,65],[180,80],[160,50],[150,45],[175,70],[190,90],[220,100]]
y = ["Fit","Fit","Not Fit","Not Fit","Fit","Not Fit","Fit"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

prediction = model.predict([[100,30]])
print(prediction)