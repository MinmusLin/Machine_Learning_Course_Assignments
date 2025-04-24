import numpy as np
from Dataset_Partitioning import Cross_Validation, Hold_out, Bootstrapping
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def get_Best_M(train_data, Ms, method, parameters):
    if method == 'Cross Validation':
        T = parameters[0]
        K = parameters[1]
        return get_CV(train_data, Ms, T, K)
    elif method == 'Hold Out':
        test_ratio = parameters[0]
        return get_HO(train_data, Ms, test_ratio)
    elif method == 'Bootstrapping':
        times = parameters[0]
        return get_B(train_data, Ms, times)

def get_CV(train_data, Ms, T, K):

    Max_Avarage_Accuracy = 0
    Best_M = Ms[0]

    for M in Ms:
        model_M = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())
        Sum_Accuracy = 0
        for n in range(T):
            np.random.shuffle(train_data)
            for k in range(K):
                train_data_k, validate_data_k = Cross_Validation(train_data, K, k+1)
                print('训练集样本数量：'+str(len(train_data_k)))
                print('验证集样本数量：'+str(len(validate_data_k)))

                model_M.fit(train_data_k[:, 1:], train_data_k[:, 0])

                boolnum = len(validate_data_k)
                boolT = 0
                output = [q for p, q in model_M.predict_proba(validate_data_k[:, 1:])]
                for i in range(boolnum):
                    if validate_data_k[i][0] == 0:
                        if output[i] < 0.5:
                            boolT += 1
                    elif validate_data_k[i][0] == 1:
                        if output[i] >= 0.5:
                            boolT += 1

                Sum_Accuracy += boolT / boolnum
                print(str(M) + "阶逻辑回归模型、第" + str(n+1) + "次、第" + str(k+1) + "折的准确率为" + str(round(100 * boolT / boolnum, 2)) + "%\n")

        Avarage_Accuracy = Sum_Accuracy / (T * K)
        print(str(M) + "阶逻辑回归模型" + str(T) + "次" + str(K) + "折交叉检验的平均准确率为" + str(round(100 * Avarage_Accuracy, 2)) + "%\n")
        if Avarage_Accuracy > Max_Avarage_Accuracy:
            Max_Avarage_Accuracy = Avarage_Accuracy
            Best_M = M
    print("最佳模型为" + str(Best_M) + "阶逻辑回归模型，其在交叉验证法验证集上的平均准确率为" + str(round(100 * Max_Avarage_Accuracy, 2)) + "%\n")
    return Best_M

def get_HO(train_data, Ms, test_ratio):
    Max_Accuracy = 0
    Best_M = Ms[0]
    train_data_k, validate_data_k = Hold_out(train_data, test_ratio)
    print('训练集样本数量：' + str(len(train_data_k)))
    print('验证集样本数量：' + str(len(validate_data_k)))
    for M in Ms:
        model_M = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())
        model_M.fit(train_data_k[:, 1:], train_data_k[:, 0])

        boolnum = len(validate_data_k)
        boolT = 0
        output = [q for p, q in model_M.predict_proba(validate_data_k[:, 1:])]
        for i in range(boolnum):
            if validate_data_k[i][0] == 0:
                if output[i] < 0.5:
                    boolT += 1
            elif validate_data_k[i][0] == 1:
                if output[i] >= 0.5:
                    boolT += 1
        Accuracy = boolT / boolnum
        print(str(M) + "阶逻辑回归模型在验证集上的准确率为" + str(round(100 * Accuracy, 2)) + "%\n")
        if Accuracy > Max_Accuracy:
            Max_Accuracy = Accuracy
            Best_M = M
    print("最佳模型为" + str(Best_M) + "阶逻辑回归模型，其在留出法验证集上的准确率为" + str(round(100 * Max_Accuracy, 2)) + "%\n")
    return Best_M

def get_B(train_data, Ms, times):
    Max_Accuracy = 0
    Best_M = Ms[0]
    train_data_k, validate_data_k = Bootstrapping(train_data, times)
    print('训练集样本数量：' + str(len(train_data_k)))
    print('验证集样本数量：' + str(len(validate_data_k)))
    for M in Ms:
        model_M = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())
        model_M.fit(train_data_k[:, 1:], train_data_k[:, 0])

        boolnum = len(validate_data_k)
        boolT = 0
        output = [q for p, q in model_M.predict_proba(validate_data_k[:, 1:])]
        for i in range(boolnum):
            if validate_data_k[i][0] == 0:
                if output[i] < 0.5:
                    boolT += 1
            elif validate_data_k[i][0] == 1:
                if output[i] >= 0.5:
                    boolT += 1
        Accuracy = boolT / boolnum
        print(str(M) + "阶逻辑回归模型在验证集上的准确率为" + str(round(100 * Accuracy, 2)) + "%\n")
        if Accuracy > Max_Accuracy:
            Max_Accuracy = Accuracy
            Best_M = M
    print("最佳模型为" + str(Best_M) + "阶逻辑回归模型，其在自助法验证集上的准确率为" + str(round(100 * Max_Accuracy, 2)) + "%\n")
    return Best_M