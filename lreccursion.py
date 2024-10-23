import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.linear_model import LinearRegression
from sklearn import metrics
iris=load_iris()
x=iris.data
y=iris.target
x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x[:, 1:], x[:, 0], test_size=0.3, random_state=1)
lin_reg=LinearRegression()
lin_reg.fit(x_reg_train,y_reg_train)
y_reg_pred = lin_reg.predict(x_reg_test)
mse = metrics.mean_squared_error(y_reg_test, y_reg_pred)
print("Linear Regression Mean Squared Error:", mse)
plt.figure()
plt.scatter(y_reg_test, y_reg_pred)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'k--', lw=2)
plt.xlabel('True Values (Sepal Length)')
plt.ylabel('Predictions (Sepal Length)')
plt.title('True vs Predicted Values for Linear Regression')
plt.show()
