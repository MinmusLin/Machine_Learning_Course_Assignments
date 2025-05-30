from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def Predict(Algorithm, test):
    algs = {
        'KNN':KNeighborsClassifier(),
        'Logistic':LogisticRegression(),
        'DecisionTree':DecisionTreeClassifier(),
        'SVM':SVC(random_state=12,kernel='linear',max_iter=200, C=100000000.)
    }

    model = algs.get(Algorithm, None)
    assert model != None, 'no such algorithm'
    model.fit(X_train, y_train)
    prediction = model.predict(test)

    return prediction

def Vis(X_test, y_test, y_pred):
    color = np.random.rand(20, 3)
    split_X1 = [X_test[y_pred == value] for value in np.unique(y_pred)]

    for one_class, class_num in zip(split_X1, np.unique(y_pred)):
        plt.scatter(one_class[:,1], one_class[:,2], c=color[class_num], label='pred '+str(class_num))

    split_X2 = [X_test[y_test == value] for value in np.unique(y_test)]

    for one_class, class_num in zip(split_X2, np.unique(y_test)):
        plt.scatter(one_class[:,1], one_class[:,2], c=color[class_num], s=100, label='true '+str(class_num), marker='x')

    plt.axis('equal')
    plt.legend()
    plt.show()

scaler = StandardScaler()
iris_dataset = load_iris()
X = iris_dataset.data
X[:, 0] = X[:, 0]*5
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, iris_dataset['target'], random_state=0)
y_pred = Predict('SVM', X_test)

print('Test set predictions:\n {}'.format(y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

Vis(X_test, y_test, y_pred)