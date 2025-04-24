import random
import numpy as np
import math

def Cross_Validation(data, fold, k):

    assert k > 0 and k <= fold, "k should in [1, fold]"

    train_data = []
    validate_data = []

    for i in range(len(data)):
        if i >= math.ceil(len(data)/fold)*(k-1) and i < math.ceil(len(data)/fold)*k and i < len(data):
            validate_data.append(data[i])
        else:
            train_data.append(data[i])

    return np.array(train_data), np.array(validate_data)

def Hold_out(data, test_ratio):
    class0 = []
    class1 = []

    for d in data:
        if d[0] == 0:
            class0.append(d)
        else:
            class1.append(d)

    train_data = []
    test_data = []

    for i in range(len(class0)):
        if i < len(class0)*test_ratio:
            test_data.append(class0[i])
        else:
            train_data.append(class0[i])

    for i in range(len(class1)):
        if i < len(class1)*test_ratio:
            test_data.append(class1[i])
        else:
            train_data.append(class1[i])

    return np.array(train_data), np.array(test_data)

def Bootstrapping(data, times):
    in_train = np.zeros(len(data))
    indexes = [random.randint(0, len(data)-1) for _ in range(times)]

    train_data = []
    test_data = []

    for index in indexes:
        train_data.append(data[index])
        in_train[index] = 1
    for index in range(len(data)):
        if in_train[index] == 0:
            test_data.append(data[index])

    return np.array(train_data), np.array(test_data)