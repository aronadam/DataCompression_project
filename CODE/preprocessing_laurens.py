import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import cvxpy as cp
from timeit import default_timer as t_rec
import csv


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


def NMSE_plot(X_in,y,n_samples=100):

    # Number of samples (rows in the fat matrix)
    M_list = [50, 100, 200, 300, 400, 500, 600, 700, 784]

    labels = np.linspace(0,9,10)

    
    NMSE_lists = [[] for _ in range(len(M_list))]
    mean_NMSE = np.zeros([9,len(labels)])
    times = np.zeros([9,len(labels)])
    for label in labels:
            
        X = X_in[y==label]
        X = X[:n_samples]
        X_index = range(len(X))
        # Compute NMSE for each M for various X's
        for i in range(len(M_list)):
            t_loop_1 = t_rec()
            for k in X_index:
                x = X[k]
                reconstructed = one_bit_cs_by_lp(M_list[i], x)
                MSE = np.mean((x - reconstructed) ** 2)
                NMSE = MSE / (np.linalg.norm(x) ** 2)
                NMSE_lists[i].append(NMSE)
                # print(f"Reconstructed X_{k} with {M_list[i]} samples")
            t_loop_2=t_rec()
            times[i,int(label)] = (t_loop_2-t_loop_1)/(k+1)
            print(f"For label {int(label)}, all reconstructions finished for A =     {M_list[i]}     x 784.\nThis operation took {round(t_loop_2-t_loop_1,2)} seconds.\n")
        # Average the NMSE for each M
        mean_NMSE[:,int(label)] = [np.mean(NMSE_list) for NMSE_list in NMSE_lists]

        # Plot the mean NMSE for each M
        plt.plot(M_list, mean_NMSE[:,int(label)], marker=">")
        plt.xlabel('Number of samples')
        plt.ylabel('NMSE (dB)')
        plt.title(f'Error propagation for pictures of digit {int(label)}')
        plt.ylim(0.0)
        plt.show(block=False)
        plt.savefig(f'C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\Images\\NMSE_{int(label)}.png')
        plt.pause(2)
        plt.clf()
        plt.close()
    mean_NMSE_overall = [np.mean(mean_NMSE) for mean_NMSE in mean_NMSE]
    with open('C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\tmp_file.txt', 'w') as f:
        csv.writer(f, delimiter=',').writerows(times)
            
    plt.plot(M_list, mean_NMSE_overall, marker=">")
    plt.xlabel('Number of samples')
    plt.ylabel('NMSE (dB)')
    plt.title(f'Overall error propagation')
    plt.ylim(0.0)
    plt.show()

def reconstruction_plots(X,y):
    reconstructed = np.zeros([10,784,5])
    original = np.zeros([784,5])
    for label in y:
        label = int(label)
        x = (X[y==label])[0]
        original[:,i] = x.copy()
        M_list = [50, 100,  500, 700, 784]
        for i in range(len(M_list)):
            reconstructed[label,:,i] = one_bit_cs_by_lp(M_list[i], x)
    




if __name__ == "__main__":
    
    mnist = pd.read_csv("C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\DATA\\mnist_test.csv", header=None).to_numpy()
    X, y = mnist[:, 1:], mnist[:, 0]
     
    t_start = t_rec()
    NMSE_plot(X,y)
    t_end = t_rec()
    print(f"Full Operation took {round((t_end-t_start),2)} seconds")

    reconstruction_plots(X,y)
