import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def fat_matrix(M, N=784):
    Z = np.random.standard_normal(size=(M, N))
    return normalize(Z, norm="l2", axis=1)


if __name__ == "__main__":

    mnist = pd.read_csv("../DATA/mnist_train.csv", header=None).to_numpy()

    X, y = mnist[:, 1:], mnist[:, 0]

    fat_25 = fat_matrix(25)

    input = np.sign(fat_25 @ X[0])
    print(input)

