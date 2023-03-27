import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize as norm
from cvxopt import solvers, matrix
import keras

def get_data(mode):
    if mode in ["train","test"]:
        return np.genfromtxt(f"C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\DATA\\mnist_{mode}.csv", delimiter=',')
    else:
        print("mode should be either 'train', or 'test'")

#creates a fat matrix with elements a_ij \in {-1,1}
def fat(m):
    n=28**2
    return np.sign(np.random.randn(m,n))

def one_bit_linprog(A,x):
    m,N = A.shape
    # obj = min x
    c_1 = matrix(np.ones([m,1]))
    # c_2 = c_1.copy()*(-1)
    A_eq = matrix(A.T)
    b_eq = matrix(x)
    A_ineq = matrix(np.eye(m))
    b_ineq = matrix(np.zeros([m,1]))
    # s.t. = y = sign(Ax), x>0 || x<0
    sol_1 = solvers.lp(c_1,G=-A_ineq,h=b_ineq,A=A_eq,b=b_eq)#,G=,h=)
    # sol_2 = solvers.lp(c_2,A=A,b=x)#,G=,h=)
    # if norm(sol_1(x),norm="l1")>norm(sol_1(x),norm="l2"):
    #     return sol_1(x)
    return sol_1(x)
    # else:
    #     return sol_2['x']
    
def one_bit_NN(A,x,m=5,l=3): # CODE STRUCTURE EXAMPLE
    model = keras.models.Sequential()
    [m,N] = A.shape
    model.add(Dense(m, input_dim=N, activation='relu'))
    for i in range(l):
        model.add(Dense(m, activation='relu'))
    model.add(Dense(N, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
# YOUR CODE HERE
    model.compile(optimizer="adam",
    loss="binary_crossentropy",
    metrics="accuracy",
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
)
    validation_split = 0.33
    epochs = 30
    batch_size = 10

    history = model.fit(X_train, y_train_one_hot, validation_split=validation_split,\
                        epochs=epochs, batch_size=batch_size, callbacks = [cp_callback])



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

    sol = one_bit_linprog(A_25,x[:,1])

    


