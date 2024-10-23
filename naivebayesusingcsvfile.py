import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
iris=pd.read_csv('iris(1).csv')
x=iris.iloc[:,:-1].values
y=iris.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))
print("Enter The Sample Data")
a=float(input("Enter Sepal Length In CM : "))
b=float(input("Enter Sepal width In CM : "))
c=float(input("Enter Petal Length In CM : "))
d=float(input("Enter Petal width In CM : "))
sample=[[a,b,c,d]]
pred=nb.predict(sample)
print(pred)
