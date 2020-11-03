# Deep Learning</h2>
# In this project, we implement a neural network.

import numpy as np
from numpy.matlib import repmat
import matplotlib

import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import time

# sys.path.insert(0, './p8/')

#get_ipython().run_line_magic('matplotlib', 'inline')



# Implementing the preprocess function.
# Training data set with zero-mean and each feature having standard deviation 1 . 

def preprocess(xTr, xTe):
    """
    Preproces the data to make the training features have zero-mean and
    standard-deviation 1
    OUPUT:
        xTr - nxd training data
        xTe - mxd testing data
    OUPUT:
        xTr - pre-processed training data
        xTe - pre-processed testing data
        s,m - standard deviation and mean of xTr
            - any other data should be pre-processed by x-> (x-m)/s
    (The size of xTr and xTe should remain unchanged)
    """
    
    ntr, d = xTr.shape
    nte, _ = xTe.shape

        
    m=np.mean(xTr)
    s=np.std(xTr)
    
    xTr=(xTr-m)/s
    xTe=(xTe-m)/s


    return xTr, xTe, s, m

#   Now implementing three transition functions: ReLU, sigmoid and tanh.

def get_transition_func(transtype):
    """
    Given the type, gets a specific transition function
    INPUT:
        transtype - "sigmoid", "tanh", "ReLU", "sin"
    OUTPUT:
        trans_func - transition function (function)
        trans_func_der - derivative of the transition function (function)

    (type must be one of the defined transition functions)
    """
    
    assert transtype in ["sigmoid", "tanh", "ReLU","sin"]
    if transtype == "sin":   
        trans_func = lambda z: np.sin(z)
        trans_func_der = lambda z: np.cos(z)

    elif transtype == "sigmoid":
        trans_func = lambda z: 1/(1 + np.exp(-z))
        trans_func_der = lambda z: np.exp(-z)/(1+np.exp(-z))**2
        
    elif transtype == "tanh":
        trans_func = lambda z: np.tanh(z)
        trans_func_der = lambda z: 1 - np.square(np.tanh(z))
        
    elif transtype == "ReLU":
        trans_func = lambda z: (np.maximum(0,z))
        trans_func_der = lambda z: np.sign(np.maximum(0,z)) 

    
    return trans_func, trans_func_der


# The gradient errors should all be very small (less than $10^{-10}$.) </p>

def numericalgradient(fun,x,e):
    dh = 0
    nx = x    # copy the weight vector
    nx += e  # perturn dimension i
    l1 = fun(nx) # compute loss
    nx -= 2*e # perturm dimension i again
    l2 = fun(nx) # compute loss
    dh = (l1 - l2)/(2*e) # the gradient is the slope of the loss
    return dh


fig, axarr = plt.subplots(1, 4, sharex=True, sharey=True)
fig.set_figwidth(15)
fig.set_figheight(3)
names = ["sigmoid","tanh","ReLU","sin"]

for idx, name in enumerate(names):
    # plot stuff
    f, delta_f = get_transition_func(name)
    x = np.arange(-5, 5, 0.1)
    axarr[idx].plot(x, f(x))
    axarr[idx].axis([-5,5,-1,1])
    axarr[idx].title.set_text(name)
    axarr[idx].grid(True)
    
    # check gradients
    print("%s gradient check at x=1: " % name, end='')
    dh = numericalgradient(f,1,1e-5)
    dy = delta_f(1)
    num = np.linalg.norm(dh-dy)
    denom = np.linalg.norm(dh+dy)
    graderror = num/denom if denom != 0 else 0
    if graderror < 1e-10:
        print("passed ", end='')
    else:
        print("FAILED ", end='')
    print('at x=-1: ', end='')
    dh2 = numericalgradient(f,-1,1e-5)
    dy2 = delta_f(-1)
    num = np.linalg.norm(dh2-dy2)
    denom = np.linalg.norm(dh2+dy2)
    graderror += num/denom if denom != 0 else 0
    if graderror < 1e-10:
        print("passed")
    else:
        print("FAILED")

# The following function randomly generates initial weights for the neural network.


def initweights(specs):
    """
    Given a specification of the neural network, output a random weight array
    INPUT:
        specs - array of length m+1
    
    OUTPUT:
        weights - array of length m, each element is a matrix
            where size(weights[i]) = (specs[i], specs[i+1])
    """
    weights = []
    for i in range(len(specs) - 1):
        weights.append(np.random.randn(specs[i], specs[i+1]))
    return weights


# Now implementing the forward pass function

def forward_pass(weights, xTr, trans_func):
    """
    INPUT:
        weights - weights (cell array of length m)
        xTr - nxd matrix (each row is an input vector)
        trans_func - transition function to apply for inner layers
    
    OUTPUTS:
        A, Z - result of forward pass (cell array of length m+1)

    """
    
    n = xTr.shape[0]
    m = len(weights)
    
    A = []
    Z = []
    A.append(xTr)
    Z.append(xTr)
    
    for i in range(m):
        A.append(np.matmul(Z[i],weights[i]))
        if ((i+1) == m):
            Z.append(A[-1])
        else:
            Z.append(trans_func(A[i+1]))

        
    return A, Z

weights=initweights([2,20,20,1])



f,delta_f=get_transition_func("sigmoid")
x=np.random.rand(100,2);
A,Z=forward_pass(weights,x,f)


# Now implementing the function to compute the loss for the network


def compute_loss(Z, yTr):
    """
    INPUT:
        Z - output of forward pass (cell array of length m+1)
        yTr - array of length n
    
    OUTPUTS:
        loss - the average squared loss obtained with Z and yTr (scalar)
    """
    
    delta = Z[-1].flatten() - yTr.flatten()
    n = len(yTr)
    loss = 0

    initial = 0
    for i in range (n):
        initial = initial + (delta[i]**2)
    loss = (1/(2*n)) * initial


    return loss

yTr = np.random.rand(100)
compute_loss(Z,yTr)


# Now implementing back propagation
def backprop(weights, A, Z, yTr, delta_f):
    """
    INPUT:
        weights - weights (cell array of length m)
        A - output of forward pass (cell array of length m+1)
        Z - output of forward pass (cell array of length m+1)
        yTr - array of length n
        delta_f - derivative of transition function to apply for inner layers
    
    OUTPUTS:
        gradient - the gradient at w (cell array of length m)
    """
    
    yTr = yTr.reshape(-1,1)
    n,_ = yTr.shape
    delta = (Z[-1].flatten() - yTr.flatten()).reshape(-1, 1)
    
    # compute gradient with back-prop
    gradients = []
    
    m = len(weights)-1
    
    gradient = np.transpose(np.dot(np.transpose(delta),Z[m]))/n
    delta = np.dot(delta, np.transpose(weights[m]))
    gradients.append(gradient)                         
    
    j = m - 1
    while (j>=0):
        gradient = np.dot(np.transpose(Z[j]), (np.multiply(delta,delta_f(A[j+1]))))/n
        delta = np.dot(np.multiply(delta,delta_f(A[j+1])), np.transpose(weights[j]))
        gradients.append(gradient)
        j = j-1
        
        
    gradients.reverse()
        
    """
        grad = np.matmul(weights(m),np.multiply(delta_f(A[m]),grad))
        gradients.append(grad)
        
        delta/n*delta_f(A[-1])
    """

    return gradients
#The following function will plot the results of prediction and the loss


def plot_results(x, y, Z, losses):
    fig, axarr = plt.subplots(1, 2)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    axarr[0].plot(x, y)
    axarr[0].plot(x, Z[-1].flatten())
    axarr[0].set_ylabel('$f(x)$')
    axarr[0].set_xlabel('$x$')
    axarr[0].legend(['Actual', 'Predicted'])

    axarr[1].semilogy(losses)
    axarr[1].title.set_text('Loss')
    axarr[1].set_xlabel('Epoch')

    plt.show()



#  We can use the gradients computed from back propagation to update the weights in each layer. 
# Three variations of gradient descent.

# Momentum

x = np.arange(0, 5, 0.1)
y = (x ** 2 + 10*np.sin(x))

x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

weights = initweights([2,200,1])
momentum = np.copy(weights)
for j in range(len(weights)):
    momentum[j] = momentum[j] * 0
    
alpha = 0.01
M = 10000
beta = 0.8

losses = np.zeros(M)
t0 = time.time()
for i in range(M):
    f, delta_f = get_transition_func("ReLU")
    A, Z = forward_pass(weights, x2d, f)
    losses[i] = compute_loss(Z,y)
    gradients = backprop(weights,A,Z,y,delta_f)
    for j in range(len(weights)):
        momentum[j] = beta * momentum[j] + (1 - beta) * gradients[j]
        weights[j] -= alpha * momentum[j]
t1 = time.time()
print('Elapsed time: %.2fs ' % (t1-t0))



plot_results(x, y, Z, losses)



# ## Custom step-size decrease schedule 

# training data
x = np.arange(0, 5, 0.1)
y = (x ** 2 + 10*np.sin(x))
x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

# transition function
f, delta_f = get_transition_func("ReLU")

# initialize weights, historical gradients, losses
weights = initweights([2,200,1])
losses = np.zeros(M)

hist_grad = np.copy(weights)
for j in range(len(weights)):
    hist_grad[j] = hist_grad[j] * 0

alpha = 0.02
M = 10000
beta = 0.8
eps = 1e-6

losses = np.zeros(M)
t0 = time.time()
for i in range(M):
    f, delta_f = get_transition_func("ReLU")
    A, Z = forward_pass(weights, x2d, f)
    losses[i] = compute_loss(Z,y)
    gradients = backprop(weights,A,Z,y,delta_f)
    for j in range(len(weights)):
        if i != 0:
            hist_grad[j] = beta * hist_grad[j] + (1 - beta) * gradients[j] ** 2
        else:
            hist_grad[j] = gradients[j] ** 2
        adj_grad = gradients[j] / (eps + np.sqrt(hist_grad[j]))
        weights[j] -= alpha * adj_grad
t1 = time.time()
print('Elapsed time: %.2fs ' % (t1-t0))


plot_results(x, y, Z, losses)


# ## Adagrad


# training data
x = np.arange(0, 5, 0.1)
y = (x ** 2 + 10*np.sin(x))
x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

# transition function
f, delta_f = get_transition_func("ReLU")

# initialize weights, historical gradients, losses
weights = initweights([2,200,1])
losses = np.zeros(M)

hist_grad = np.copy(weights)
for j in range(len(weights)):
    hist_grad[j] = hist_grad[j] * 0

alpha = 0.02
M = 10000
beta = 0.8
eps = 1e-6

losses = np.zeros(M)
t0 = time.time()
for i in range(M):
    f, delta_f = get_transition_func("ReLU")
    A, Z = forward_pass(weights, x2d, f)
    losses[i] = compute_loss(Z,y)
    gradients = backprop(weights,A,Z,y,delta_f)
    for j in range(len(weights)):
        hist_grad[j] += gradients[j] ** 2
        adj_grad = gradients[j] / (eps + np.sqrt(hist_grad[j]))
        weights[j] -= alpha * adj_grad
t1 = time.time()
print('Elapsed time: %.2fs ' % (t1-t0))


plot_results(x, y, Z, losses)


