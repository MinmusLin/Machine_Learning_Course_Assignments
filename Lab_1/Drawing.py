import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def drawing_data(data, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)

def drawing_model(data, model, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)
    xx, yy = np.meshgrid(np.arange(-8, 6, 0.01),
                         np.arange(-8, 6, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)

def drawing_models(models, test_data, Ms, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in test_data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    xx, yy = np.meshgrid(np.arange(-8, 6, 0.01), np.arange(-8, 6, 0.01))
    for M_model in models:
        Z = M_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k', linewidths=1.5)

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)

def computer_table(Label, Output):
    pairs = np.array([(x, y) for x, y in zip(Label, Output)])
    sorted_pairs = pairs[pairs[:, 1].argsort()]

    TPs = []
    FNs = []
    FPs = []
    TNs = []

    TP = 0
    FN = np.count_nonzero(Label == 0)
    FP = 0
    TN = np.count_nonzero(Label == 1)

    for i in range(len(sorted_pairs)):
        if sorted_pairs[i][0] == 0:
            FN -= 1
            TP += 1
            TPs.append(TP)
            FNs.append(FN)
            FPs.append(FP)
            TNs.append(TN)
        else:
            TN -= 1
            FP += 1
            TPs.append(TP)
            FNs.append(FN)
            FPs.append(FP)
            TNs.append(TN)

    TPs = np.array(TPs)
    FNs = np.array(FNs)
    FPs = np.array(FPs)
    TNs = np.array(TNs)

    return TPs, FNs, FPs, TNs

def computer_table2(Label, Output):
    pairs = np.array([(x, y) for x, y in zip(Label, Output)])
    sorted_pairs = pairs[pairs[:, 1].argsort()]

    TPs = []
    FNs = []
    FPs = []
    TNs = []

    TP = np.count_nonzero(Label == 1)
    FN = 0
    FP = np.count_nonzero(Label == 0)
    TN = 0

    for i in range(len(sorted_pairs)):
        if sorted_pairs[i][0] == 0:
            FP -= 1
            TN += 1
            TPs.append(TP)
            FNs.append(FN)
            FPs.append(FP)
            TNs.append(TN)
        else:
            TP -= 1
            FN += 1
            TPs.append(TP)
            FNs.append(FN)
            FPs.append(FP)
            TNs.append(TN)

    TPs = np.array(TPs)
    FNs = np.array(FNs)
    FPs = np.array(FPs)
    TNs = np.array(TNs)

    return TPs, FNs, FPs, TNs

def compute_PR(Label, Output):
    TPs, FNs, FPs, TNs = computer_table(Label, Output)

    Ps = TPs / (TPs + FPs)
    Rs = TPs / (TPs + FNs)

    return Ps, Rs

def compute_ROC(Label, Output):
    TPs, FNs, FPs, TNs = computer_table(Label, Output)

    TPRs = TPs / (TPs + FNs)
    FPRs = FPs / (TNs + FPs)

    return TPRs, FPRs

def drawing_PR(Label, Output, title):
    Ps, Rs = compute_PR(Label, Output)

    plt.figure(title)
    plt.plot(Rs, Ps,label=title)
    plt.plot([1, 0], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

def drawing_ROC(Label, Output, title):
    TPRs, FPRs = compute_ROC(Label, Output)

    plt.figure(title)
    plt.plot(FPRs, TPRs, label=title)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

def drawing_PRs(outputs, test_data, Ms, title):
    plt.figure(title)
    for output, M in zip(outputs, Ms):
        Ps, Rs = compute_PR(test_data[:, 0], output)
        plt.plot(Rs, Ps, label= str(M) + '阶逻辑回归分类器PR曲线')

    plt.plot([1, 0], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

def drawing_ROCs(outputs, test_data, Ms, title):
    plt.figure(title)
    for output, M in zip(outputs, Ms):
        TPRs, FPRs = compute_ROC(test_data[:, 0], output)
        plt.plot(FPRs, TPRs, label= str(M) + '阶逻辑回归分类器ROC曲线')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()