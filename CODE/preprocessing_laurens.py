import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import cvxpy as cp


def fat_matrix(M, N=784):
    Z = np.random.standard_normal(size=(M, N))
    return normalize(Z, norm="l2", axis=1)


def one_bit_cs_by_lp(M_samples, x):
    A = fat_matrix(M_samples)
    M, N = A.shape

    y = np.sign(A @ x)

    u = cp.Variable(N)
    x = cp.Variable(N)

    constraints = [x <= u]
    constraints += [x >= -u]
    constraints += [np.diag(y) @ A @ x >= np.zeros(M)]
    constraints += [cp.sum(np.diag(y) @ A @ x) >= M*255]

    prob = cp.Problem(cp.Minimize(cp.sum(u)), constraints)

    prob.solve()

    return np.array(x.value)


if __name__ == "__main__":

    mnist = pd.read_csv("../DATA/mnist_train.csv", header=None).to_numpy()

    X, y = mnist[:, 1:], mnist[:, 0]

    # Image to reconstruct
    x = X[0]

    # Reconstruct using various sample sizes
    reconstructed_list = []
    for M in [25, 100, 200, 400, 500]:
        reconstructed = one_bit_cs_by_lp(M, x)
        reconstructed_list.append(reconstructed)

    # Plot result
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.suptitle("1-bit Compressed Sensing by LP")
    ax1.imshow(reconstructed_list[0].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    ax1.set_title("M = 25")
    ax2.imshow(reconstructed_list[1].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    ax2.set_title("M = 100")
    ax3.imshow(reconstructed_list[2].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    ax3.set_title("M = 200")
    ax4.imshow(reconstructed_list[3].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    ax4.set_title("M = 400")
    ax5.imshow(reconstructed_list[4].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    ax5.set_title("M = 500")
    ax6.imshow(x.reshape(28,28), cmap='gray', vmin=0, vmax=255)
    ax6.set_title("Original image")
    plt.show()
