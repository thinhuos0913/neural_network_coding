import numpy as np
import matplotlib.pyplot as plt
import h5py

# Load data
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Initialize parameters:
def initialize_parameters(n0, n1, n2):
    """ Initialize W1, b1, W2, b2
    n0: dimension of input data
    n1: number of hidden unit
    n2: number of output unit = number of classes
    """
    W1 = 0.01*np.random.randn(n1, n0)
    b1 = np.zeros((n1,1))
    W2 = 0.01*np.random.randn(n2, n1)
    b2 = np.zeros((n2,1))
    return W1, b1, W2, b2

# Activation functions
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))

def ReLu(z):
    return np.maximum(z,0)

def ReLu_grad(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def LeakyReLu(x,eta=0.01):
    return abs(eta*x)*(x>0)

def dLeakyReLu(x, eta=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = eta
    return dx

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each ROW of Z is a set of scores.
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

# Loss function
def cross_entropy_loss(Yhat, y):
    """
    Y-hat: a numpy array of shape (N-points, n-Classes) --- predicted output
    y: a numpy array of shape (N-points) --- ground truth.

    """
    id0 = range(Yhat.shape[0])
    return -np.mean(np.log(Yhat[id0, y]))

def loss_function(Yhat, y):
    """
    Y-hat: a numpy array of shape (N-points, n-Classes) --- predicted output
    y: a numpy array of shape (N-points) --- ground truth.
    
    """
    # id0 = range(Yhat.shape[0])
    # return -np.mean(np.log(Yhat[id0, y])) # cross_entropy loss function
    return -np.mean(y*np.log(Yhat) + (1-y)*np.log(1-Yhat)) # logistic loss function

# Regularization
def regularized_loss(Yhat, y, lambd):
    m = y.shape[1]
    loss = -np.mean(y*np.log(Yhat) + (1-y)*np.log(1-Yhat))
    L2_reg = lambd/(2*m)*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    cost = loss + L2_reg
    
    return cost

# Prediction
def predict(X, W1, b1, W2, b2, activation):
    """
    Suppose the network has been trained, predict class of new points.
    X: data matrix, each ROW is one data point.
    W1, b1, W2, b2: trained weight matrices and biases
    
    """
    Z1 = np.dot(W1,X) + b1 # shape (N, d1)
    if activation == 'relu':
        A1 = ReLu(Z1)
    elif activation == 'sigmoid':
        A1 = sigmoid(Z1) # shape (N, d1)
    else:
        A1 = LeakyReLu(Z1)
    Z2 = np.dot(W2,A1) + b2 # shape (N, d2)
    return sigmoid(Z2)

# Decide predicted class is 0 or 1
def separate(p,x):
    p = p.reshape(x.shape[1],)
    for i in range(len(p)):
        if p[i] >= 0.5: 
            p[i] = 1
        else:
            p[i] = 0
    return p

# Evaluation
def evaluate(p,y):
    y = y.reshape(y.shape[1],)
    acc = 100*np.mean(p == y)
    print("Accuracy = ", acc)
    return acc

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (30.0, 30.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
