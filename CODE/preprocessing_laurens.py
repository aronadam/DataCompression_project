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
    constraints += [cp.sum(np.diag(y) @ A @ x) >= M*100]

    prob = cp.Problem(cp.Minimize(cp.sum(u)), constraints)

    prob.solve()

    return np.array(x.value)


def progression_plot(x):

    # Reconstruct using various sample sizes
    reconstructed_list = []
    for M in [25, 100, 300, 500, 700]:
        reconstructed = one_bit_cs_by_lp(M, x)
        reconstructed_list.append(reconstructed)

    # Plot result
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.suptitle("1-bit Compressed Sensing by LP")
    ax1.imshow(reconstructed_list[0].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    ax1.set_title("M = 25")
    ax2.imshow(reconstructed_list[1].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    ax2.set_title("M = 100")
    ax3.imshow(reconstructed_list[2].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    ax3.set_title("M = 300")
    ax4.imshow(reconstructed_list[3].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    ax4.set_title("M = 500")
    ax5.imshow(reconstructed_list[4].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    ax5.set_title("M = 700")
    ax6.imshow(x.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    ax6.set_title("Original image")
    plt.show()


def NMSE_plot(X):

    # Number of samples (rows in the fat matrix)
    M_list = [50, 100, 200, 300, 400, 500, 600, 700, 784]

    X_index = range(len(X))

    NMSE_lists = [[] for _ in range(len(M_list))]

    # Compute NMSE for each M for various X's
    for i in range(len(M_list)):
        for k in X_index:
            x = X[k]
            reconstructed = one_bit_cs_by_lp(M_list[i], x)
            MSE = np.mean((x - reconstructed) ** 2)
            NMSE = MSE / (np.linalg.norm(x) ** 2)
            NMSE_lists[i].append(NMSE)
            print(f"Reconstructed X_{k} with {M_list[i]} samples")

    # Average the NMSE for each M
    mean_NMSE = [np.mean(NMSE_list) for NMSE_list in NMSE_lists]

    # Plot the mean NMSE for each M
    plt.plot(M_list, mean_NMSE, marker=">")
    plt.ylim(0.0)
    plt.show()


if __name__ == "__main__":

    mnist = pd.read_csv("../DATA/mnist_train.csv", header=None).to_numpy()

    X, y = mnist[:, 1:], mnist[:, 0]

    NMSE_plot(X[0:5])
