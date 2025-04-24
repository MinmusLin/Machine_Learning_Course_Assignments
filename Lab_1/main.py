import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning

from Dataset_Partitioning import Hold_out, Bootstrapping
from Drawing import drawing_data, drawing_model, drawing_PR, drawing_ROC, drawing_models, drawing_PRs, drawing_ROCs
from Dataset import dataset
from Evaluation import get_Best_M

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N = 2000
Ms = [1, 2, 5]
T = 3
K = 5

data = dataset(N)

train_data, test_data = Hold_out(data, 0.5)
print('训练集样本数量：'+str(len(train_data)))
print('测试集样本数量：'+str(len(test_data)))

best_M = 2
best_model = make_pipeline(PolynomialFeatures(degree=best_M), LogisticRegression())
best_model.fit(train_data[:, 1:], train_data[:, 0])

output = [q for p, q in best_model.predict_proba(test_data[:, 1:])]

boolnum = len(test_data)
boolT = 0
for i in range(boolnum):
    if test_data[i][0] == 0:
        if output[i] < 0.5:
            boolT += 1
    elif test_data[i][0] == 1:
        if output[i] >= 0.5:
            boolT += 1

Accuracy = boolT / boolnum
print(str(best_M) + '阶逻辑回归分类器在测试集上的准确率：' + str(round(100 * Accuracy, 2))+'%')

plt.show()