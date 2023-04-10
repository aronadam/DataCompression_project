import math
import random
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

    # Normalize x
    # norm = np.linalg.norm(x)
    # x = x / norm

    y = np.sign(A @ x)  # + np.random.normal(0.0, 0.01, M))

    u = cp.Variable(N)
    x_hat = cp.Variable(N)

    constraints = [x_hat <= u]
    constraints += [x_hat >= -u]
    constraints += [np.diag(y) @ A @ x_hat >= np.zeros(M)]
    constraints += [cp.sum(np.diag(y) @ A @ x_hat) >= 255*M]

    prob = cp.Problem(cp.Minimize(cp.sum(u)), constraints)

    prob.solve()

    return np.array(x_hat.value)


def progression_plot(x):

    # Reconstruct using various sample sizes
    reconstructed_list = []
    M_list = [25, 100, 300, 500, 700]
    for M in M_list:
        reconstructed = one_bit_cs_by_lp(M, x)
        reconstructed_list.append(reconstructed)

    # Plot result
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.suptitle("1-bit Compressed Sensing by LP")
    ax1.imshow(reconstructed_list[0].reshape(28, 28), cmap='gray_r', vmin=min(reconstructed_list[0]), vmax=max(reconstructed_list[0]))
    ax1.set_title(f"M = {M_list[0]}")
    ax2.imshow(reconstructed_list[1].reshape(28, 28), cmap='gray_r', vmin=min(reconstructed_list[1]), vmax=max(reconstructed_list[1]))
    ax2.set_title(f"M = {M_list[1]}")
    ax3.imshow(reconstructed_list[2].reshape(28, 28), cmap='gray_r', vmin=min(reconstructed_list[2]), vmax=max(reconstructed_list[2]))
    ax3.set_title(f"M = {M_list[2]}")
    ax4.imshow(reconstructed_list[3].reshape(28, 28), cmap='gray_r', vmin=min(reconstructed_list[3]), vmax=max(reconstructed_list[3]))
    ax4.set_title(f"M = {M_list[3]}")
    ax5.imshow(reconstructed_list[4].reshape(28, 28), cmap='gray_r', vmin=min(reconstructed_list[4]), vmax=max(reconstructed_list[4]))
    ax5.set_title(f"M = {M_list[4]}")
    ax6.imshow(x.reshape(28, 28), cmap='gray_r', vmin=0, vmax=255)
    ax6.set_title("Original image")
    plt.show()


def progression_plot_extended(X, y):

    # Get the first occurrence of each number
    numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    indexes = []
    for n in numbers:
        indexes.append(np.where(y == n)[0][0])
    X = X[indexes]

    # Reconstruct using various sample sizes
    M_list = [25, 100, 200, 358, 500, 0]
    fig, axs = plt.subplots(len(M_list), len(X))
    fig.suptitle("1-bit Compressed Sensing by LP")

    for M, row in zip(M_list, axs):
        for i, ax in zip(numbers, row):
            x = X[i]
            if M == 0:
                ax.imshow(x.reshape(28, 28), cmap='gray_r', vmin=min(x), vmax=max(x))
                if i == 0:
                    ax.set_ylabel("Original")
                    ax.xaxis.set_visible(False)
                    ax.tick_params(left=False, labelleft=False)
                else:
                    ax.axis('off')
            else:
                reconstructed = one_bit_cs_by_lp(M, x)
                ax.imshow(reconstructed.reshape(28, 28), cmap='gray_r', vmin=min(reconstructed), vmax=max(reconstructed))
                if i == 0:
                    ax.set_ylabel(f"M = {M}")
                    ax.xaxis.set_visible(False)
                    ax.tick_params(left=False, labelleft=False)
                else:
                    ax.axis('off')

            print(f"Plotting {i} with {M} samples")
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

            x_normalized = x / np.linalg.norm(x)
            reconstructed_normalized = reconstructed / np.linalg.norm(reconstructed)

            MSE = np.linalg.norm(x - reconstructed) ** 2
            # NMSE = np.linalg.norm(x_normalized - reconstructed_normalized) ** 2
            NMSE = np.mean(np.square(x_normalized - reconstructed_normalized))

            NMSE_lists[i].append(NMSE)
            print(f"Reconstructed X_{k} with {M_list[i]} samples")

    # Average the NMSE for each M
    mean_NMSE = [np.mean(NMSE_list) for NMSE_list in NMSE_lists]

    # Plot the mean NMSE for each M
    plt.plot(M_list, mean_NMSE, marker=">")
    plt.ylim(0.0)
    plt.show()


def sparsity_vs_measurments_plot():
    S = range(1, 785)
    N = 784
    M = [s * math.log2(N / s) for s in S]
    plt.plot(S, M)
    plt.xlabel("Sparsity level s")
    plt.ylabel("m = s log2(N / s))")
    plt.title("Number of measurements needed given sparsity level (N = 784)")
    plt.show()


if __name__ == "__main__":

    mnist_train = pd.read_csv("../DATA/mnist_train.csv", header=None).to_numpy()
    mnist_test = pd.read_csv("../DATA/mnist_test.csv", header=None).to_numpy()

    X_train, y_train = mnist_train[:, 1:], mnist_train[:, 0]
    X_test, y_test = mnist_test[:, 1:], mnist_test[:, 0]
