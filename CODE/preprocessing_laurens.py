import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import cvxpy as cp
from timeit import default_timer as t_rec
import csv
from sklearn.metrics import average_precision_score as MAP_score


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


def NMSE_plot(X_in,y,n_samples=50):

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
        with open(f'C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\times_{int(label)}.txt', 'w') as f:
            csv.writer(f, delimiter=',').writerows(times)

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
            
    plt.plot(M_list, mean_NMSE_overall, marker=">")
    plt.xlabel('Number of samples')
    plt.ylabel('NMSE (dB)')
    plt.title(f'Overall error propagation')
    plt.ylim(0.0)
    plt.show()

def MAP_plot(X_in,y,n_samples=2):

    # Number of samples (rows in the fat matrix)
    M_list = [50, 100, 200, 300, 400, 500, 600, 700, 784]

    labels = np.linspace(0,9,10)

    
    MAP_list = [[] for _ in range(len(M_list))]
    mean_MAP = np.zeros([9,len(labels)])
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
                print(type(reconstructed))
                print(reconstructed)
                reconstructed[reconstructed<128]=0
                reconstructed[reconstructed>=128]=1
                x[x<128] =0
                x[x>=128] =1
                # print(reconstructed)
                MAP = MAP_score(x,reconstructed)
                # NMSE = MSE / (np.linalg.norm(x) ** 2)
                # convert input to 1 bit input before 1B map comparison
                MAP_list[i].append(MAP)
                # print(f"Reconstructed X_{k} with {M_list[i]} samples")
            t_loop_2=t_rec()
            # times[i,int(label)] = (t_loop_2-t_loop_1)/(k+1)
            print(f"For label {int(label)}, all reconstructions finished for A =     {M_list[i]}     x 784.\nThis operation took {round(t_loop_2-t_loop_1,2)} seconds.\n")
        # Average the NMSE for each M
        mean_MAP[:,int(label)] = [np.mean(MAP_list) for MAP_list in MAP_list]
        # with open(f'C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\times_{int(label)}.txt', 'w') as f:
        #     csv.writer(f, delimiter=',').writerows(times)

        # Plot the mean NMSE for each M
        plt.plot(M_list, mean_MAP[:,int(label)], marker=">")
        plt.xlabel('Number of samples')
        plt.ylabel('MAP score')
        plt.title(f'MAP score propagation for pictures of digit {int(label)}')
        plt.ylim(0.0)
        plt.show(block=False)
        plt.savefig(f'C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\Images\\MAP_{int(label)}.png')
        plt.pause(3)
        plt.clf()
        plt.close()
    MAP_overall = [np.mean(mean_MAP) for mean_MAP in mean_MAP]
            
    plt.plot(M_list, MAP_overall, marker=">")
    plt.xlabel('Number of samples')
    plt.ylabel('NMSE (dB)')
    plt.title(f'Overall MAP propagation')
    plt.ylim(0.0)
    plt.show()

def compute_avg_sparsity(X_in,y,n_samples=1):

    # Number of samples (rows in the fat matrix)
    labels = np.linspace(0,9,10)
    avg_s = np.zeros([10,1])
    for label in labels:
        sparsity = np.zeros([n_samples,1])
        X = X_in[y==label]
        X = X[:n_samples]
        X_index = range(len(X))
        # Compute NMSE for each M for various X's
        
        for k in X_index:
            x = X[k]
            sparsity[k] = np.count_nonzero(x==0)

        avg_s[int(label)] = np.sum(sparsity)/len(X)
        print(f"Average sparsity level for number {int(label)}: {(int(avg_s[int(label)]))} out of 784\nLeaves {(int(avg_s[int(label)])-784)} nonzeros")
    overall_average = int(np.average(avg_s))
    print(f"Overall average sparsity level: {overall_average} Out of 784.\nLeaves on average {(784-overall_average)} nonzero entries")






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

def make_sim_times_plot(path):
    data = np.genfromtxt(path, delimiter=',')
    M_list = np.array([50, 100, 200, 300, 400, 500, 600, 700, 784])
    sizes = data.shape[0]
    labels = data.shape[1]
    legends=[]
    for i in range(labels):
        maximum = np.max(data[:,i])
        plt.plot(M_list,data[:,i]/maximum,linewidth=".5")
        plt.title(F"Normalized Simulation times over number of measurements")
        legends.append(f"{i}, norm = {round(maximum,2)}")
    plt.legend(legends)
    plt.xlabel("Number of measurements")
    plt.ylabel(f"Normalized simulation time per measurement [s]")
    plt.show(block=False)
    plt.savefig(f'C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\Images\\sim_times_1.png')
    plt.pause(3)
    plt.clf()
    plt.close()

    mean_times = np.sum(data,axis=1)
    plt.plot(M_list,mean_times,color="red")
    plt.title(F"Mean simulation times over different number of measurements for all labels")
    plt.xlabel("Number of measurements")
    plt.ylabel("Simulation time per measurement [s]")
    plt.show(block=False)
    plt.savefig(f'C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\Images\\sim_times_2.png')
    plt.pause(3)
    plt.clf()
    plt.close()

if __name__ == "__main__":

    mnist = pd.read_csv("C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\DATA\\mnist_train.csv", header=None).to_numpy()
    X, y = mnist[:, 1:], mnist[:, 0]

    t_start = t_rec()
    MAP_plot(X,y)
    path = "C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\CODE\\times_9.txt"
    # make_sim_times_plot(path)
    # compute_avg_sparsity(X,y,50)

    t_end = t_rec()
    print(f"Full Operation took {round((t_end-t_start),2)} seconds")

    # reconstruction_plots(X,y)
