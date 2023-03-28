import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize as norm
from cvxopt import solvers, matrix
# import keras

def get_data(mode):
    if mode in ["train","test"]:
        return np.genfromtxt(f"C:\\Users\\arona\\OneDrive\\Bureaublad\\boeken\\1_DC\\DataCompression_project\\DATA\\mnist_{mode}.csv", delimiter=',')
    else:
        print("mode should be either 'train', or 'test'")

#creates a fat matrix with elements a_ij \in {-1,1}
def fat(m):
    return norm(np.random.randn(m,28**2),norm="l2",axis=1)

def one_bit_linprog(A,y):
    m,N = A.shape

    c = matrix(np.concatenate([np.zeros([N,1]),np.ones([N,1])]))
    A_ineq_1 = np.block([[np.eye(N), -np.eye(N)],[-np.eye(N),-np.eye(N)]])
    b_ineq_1 = np.zeros([2*N,1])
    A_ineq_2 = np.block([-np.diag(y)@A , np.zeros([m,N])])
    b_ineq_2 = np.zeros([m,1])
    A_ineq_3 = np.block([-y.T@A , np.zeros([1,N])])
    b_ineq_3 = np.array([-m])

    A_ineq = matrix(np.concatenate([A_ineq_1,A_ineq_2,A_ineq_3],axis=0))
    b_ineq = matrix(np.block([[b_ineq_1],[b_ineq_2],[b_ineq_3]]))
    # solvers.lp(c,G=A_ineq,h=b_ineq)
    sol_1 = solvers.lp(c,G=A_ineq,h=b_ineq,)
    ret = np.array(sol_1['x'])
    return ret
    
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
    # y = Data[:,1].T   # 1 x 60.000
    s = 28
    # Create fat matrices
    A_25 = fat(25)  #m x 784
    A_100 = fat(100)
    A_200 = fat(200)
    A_500 = fat(500)
    # Create 1-bit Inputs
    y_25 =  np.sign(fat(25)@x)
    y_100 = np.sign(fat(100)@x)
    y_200 = np.sign(fat(200)@x)
    y_500 = np.sign(fat(500)@x)

    index = 1
    sol_1 = one_bit_linprog(A_500,y_500[:,index])
    sol_2 = one_bit_linprog(A_100,y_100[:,index])
    sol_3 = one_bit_linprog(A_200,y_200[:,index])
    sol_4 = one_bit_linprog(A_25,y_25[:,index])

    sol = (sol_1+sol_2+sol_3+sol_4)
    reconstruction = sol_2[:s*s].reshape([s,s])

    plt.imshow(reconstruction,cmap="Greys")
    plt.show()
    np.sum((x[:,index]-sol)**2)
