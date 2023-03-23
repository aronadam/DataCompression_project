import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

def get_data(mode):
    if mode in ["train","test"]:
        return np.genfromtxt(f"C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\DATA\\mnist_{mode}.csv", delimiter=',')
    else:
        print("mode should be either 'train', or 'test'")

def fat(m):
    n=28**2
    return np.random.randn(m,n)

if __name__=='__main__':
######################## Preprocessing
    # Get data
    Data = get_data("train")
    # Separate Data from labels
    x = Data[:,1:].T  # 786 x 60.000
    y = Data[:,1].T   # 1 x 60.000

    # Create fat matrices
    A_25 = fat(25)  #m x 784
    A_100 = fat(100)
    A_200 = fat(200)
    A_500 = fat(500)
    # Create 1-bit Inputs
    y_25 = np.sign(A_25@x)
    y_100 = np.sign(A_100@x)
    y_200 = np.sign(A_200@x)
    y_500 = np.sign(A_500@x)
