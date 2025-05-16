from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def drawing_loss(gammas):
    train_loss = []
    test_loss = []
    train_bias = []
    test_bias = []
    train_variance = []
    test_variance = []
    for gamma in gammas:
        model = SVC(gamma=gamma)
        model.fit(X_train, y_train)
        train_loss.append(1-model.score(X_train, y_train))
        test_loss.append(1-model.score(X_test, y_test))
        train_bias.append(np.mean(abs(model.predict(X_train) - y_train)))
        test_bias.append(np.mean(abs(model.predict(X_test) - y_test)))
        train_variance.append(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
        test_variance.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    plt.figure("SVM_LOSS")
    gammas = [f"1e{k-5}" for k in range(10)]
    plt.plot(gammas, train_loss, color='b', label='train loss')
    plt.plot(gammas, test_loss, color='r', label='test loss')
    plt.xlabel('gamma')
    plt.ylabel('LOSS')
    plt.title("SVM_LOSS")
    plt.legend()

    plt.figure("SVM Bias and Variance")
    gammas = [f"1e{k-5}" for k in range(10)]
    plt.plot(gammas, test_bias, color='r', label='test bias')
    plt.plot(gammas, test_variance, color='g', label='test variance')
    plt.plot(gammas, test_loss, color='b', label='test loss')
    plt.xlabel('gamma')
    plt.ylabel('error')
    plt.title("SVM Bias and Variance")
    plt.legend()

def drawing_PR(y_test, y_scores):
    precision = dict()
    recall = dict()
    for i in range(len(digits.target_names)):
        precision[i], recall[i], _ = precision_recall_curve((y_test == i), y_scores[:, i])

    plt.figure('Precision-Recall curve')
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green', 'yellow', 'purple', 'blue']
    for i, color in zip(range(len(digits.target_names)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label='class {}'.format(digits.target_names[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

def drawing_ROC(y_test, y_scores):
    fpr = dict()
    tpr = dict()
    for i in range(len(digits.target_names)):
        fpr[i], tpr[i], _ = roc_curve((y_test == i), y_scores[:, i])

    plt.figure('Receiver Operating Characteristic (ROC)')
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green', 'yellow', 'purple', 'blue']
    for i, color in zip(range(len(digits.target_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='class {}'.format(digits.target_names[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
gammas = [1e-5 * 10**i for i in range(10)]

drawing_loss(gammas)

best_Accuracy = 0
for gamma in gammas:
    model = SVC(gamma=gamma)
    y_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'Model: {model}, Mean Accuracy: {y_scores.mean()}, Std: {y_scores.std()}')
    if y_scores.mean() > best_Accuracy:
        best_Accuracy = y_scores.mean()
        best_gamma = gamma

best_model = SVC(gamma=best_gamma, probability=True)
best_model.fit(X_train, y_train)

y_scores = best_model.predict_proba(X_test)

drawing_PR(y_test, y_scores)
drawing_ROC(y_test, y_scores)

plt.show()