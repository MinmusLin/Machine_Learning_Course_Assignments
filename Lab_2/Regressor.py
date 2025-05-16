from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
import numpy as np

def mse(y_true, y_pred):
    n = len(y_true)
    mse = 0
    for i in range(n):
        mse += (y_true[i] - y_pred[i]) ** 2
    mse /= n
    return mse

iris_dataset = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

model1 = linear_model.LinearRegression()
model1.fit(X_train, y_train)

model2 = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.01, max_iter=1000, learning_rate='constant', eta0=0.01)
model2.fit(X_train, y_train)

MSE1 = mean_squared_error(y_test, model1.predict(X_test))
MSE2 = mean_squared_error(y_test, model2.predict(X_test))

print(MSE1, MSE2)

plt.scatter(X_test[:,0], y_test, color='blue')
min_index = np.argmin(X_test[:, 0])
max_index = np.argmax(X_test[:, 0])
plt.plot([X_test[min_index, 0], X_test[max_index, 0]], model1.predict([X_test[min_index], X_test[max_index]]), color='red', label='least squares')
plt.plot([X_test[min_index, 0], X_test[max_index, 0]], model2.predict([X_test[min_index], X_test[max_index]]), color='black', label='gradient descent')
plt.show()