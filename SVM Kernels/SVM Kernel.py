
import numpy as np
from numpy.matlib import repmat
import sys
import time
from cvxpy import *
import l2distance
import visclassifier

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import linregress

import pylab
from matplotlib.animation import FuncAnimation

get_ipython().run_line_magic('matplotlib', 'notebook')



# Iimplementing a linear support vector machine and one operating in kernel space. 

def genrandomdata(n=100,b=0.):
    # generate random data and linearly separagle labels
    xTr = np.random.randn(n, 2)
    # defining random hyperplane
    w0 = np.random.rand(2, 1)
    # assigning labels +1, -1 labels depending on what side of the plane they lie on
    yTr = np.sign(np.dot(xTr, w0)+b).flatten()
    return xTr, yTr

def primalSVM(xTr, yTr, C=1):
    """
    function (classifier,w,b) = primalSVM(xTr,yTr;C=1)
    constructs the SVM primal formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (n)
        C     | the SVM regularization parameter
    
    Output:
        fun   | usage: predictions=fun(xTe); predictions.shape = (n,)
        wout  | the weight vector calculated by the solver
        bout  | the bias term calculated by the solver
    """
    N, d = xTr.shape
    y = yTr.flatten()
    
    
    w = Variable (d)
    b = Variable()
    slack = Variable(N)
 
    constraints = [multiply(y, (xTr*w)+b) >= 1 - slack, slack >= 0]
    objective = sum_squares(w) + C*sum(slack)
    prob = Problem(Minimize(objective), constraints)
    prob.solve()
              
            
    wout = w.value
    bout = b.value
    
    fun = lambda x: x.dot(wout) + bout
    return fun, wout, bout
    
    



def arrayify(x):
    """flattens and converts to numpy"""
    return np.array(x).flatten()



xTr,yTr=genrandomdata()
fun,w,b=primalSVM(xTr,yTr,C=10)
visclassifier.visclassifier(fun,xTr,yTr,w=w,b=b)


err=np.mean(arrayify(np.sign(fun(xTr)))!=yTr)
print("Training error: %2.1f%%" % (err*100))



def updateboundary():
    global w,b,Xdata,ldata,stepsize

    _, w_pre, b_pre = primalSVM(np.transpose(Xdata),np.array(ldata),C=10)
    w = np.array(w_pre).reshape(-1)
    b = b_pre
    stepsize+=1

def updatescreen():
    global w,b,ax,line 
    q=-b/(w**2).sum()*w;
    if line==None:
        line, = ax.plot([q[0]-w[1],q[0]+w[1]],[q[1]+w[0],q[1]-w[0]],'b--')
    else:
        line.set_ydata([q[1]+w[0],q[1]-w[0]])
        line.set_xdata([q[0]-w[1],q[0]+w[1]])
    
def animate(i):
    if len(ldata)>0 and (min(ldata)+max(ldata)==0):
        if stepsize<1000:
            updateboundary()
            updatescreen();
    
def onclick(event):
    global Xdata, stepsize  
    if event.key == 'shift': # add positive point
        ax.plot(event.xdata,event.ydata,'or')
        label=1
    else: # add negative point
        ax.plot(event.xdata,event.ydata,'ob')
        label=-1    
    pos=np.array([[event.xdata],[event.ydata]])
    ldata.append(label);
    Xdata=np.hstack((Xdata,pos))
    stepsize=1;


Xdata=pylab.rand(2,0)
ldata=[]
w=[]
b=[]
line=None
stepsize=1;
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(0,1)
plt.ylim(0,1)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
ani = FuncAnimation(fig, animate,pylab.arange(1,100,1),interval=10);


# Spiral data set

def spiraldata(N=300):
    r = np.linspace(1,2*np.pi,N)
    xTr1 = np.array([np.sin(2.*r)*r, np.cos(2*r)*r]).T
    xTr2 = np.array([np.sin(2.*r+np.pi)*r, np.cos(2*r+np.pi)*r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1])*0.2
    
    xTe = xTr[::2,:]
    yTe = yTr[::2]
    xTr = xTr[1::2,:]
    yTr = yTr[1::2]
    
    return xTr,yTr,xTe,yTe



xTr,yTr,xTe,yTe=spiraldata()
plt.scatter(xTr[yTr == 1, 0], xTr[yTr == 1, 1], c='b')
plt.scatter(xTr[yTr != 1, 0], xTr[yTr != 1, 1], c='r')
plt.legend(["+1","-1"])
plt.show()



fun,w,b=primalSVM(xTr,yTr,C=10)
visclassifier.visclassifier(fun,xTr,yTr,w=[],b=0)
err=np.mean(arrayify(np.sign(fun(xTr)))!=yTr)
print("Training error: %2.1f%%" % (err*100))


# Implementing a kernelized SVM

def computeK(kerneltype, X, Z, kpar=0):
    """
    function K = computeK(kernel_type, X, Z)
    computes a matrix K such that Kij=k(x,z);
    for three different function linear, rbf or polynomial.
    
    Input:
    kerneltype: either 'linear','polynomial','rbf'
    X: n input vectors of dimension d (nxd);
    Z: m input vectors of dimension d (mxd);
    kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
    
    OUTPUT:
    K : nxm kernel matrix
    """
    assert kerneltype in ["linear","polynomial","poly","rbf"], "Kernel type %s not known." % kerneltype
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"
    
    
    def l2distance(X, Z=None):
        if Z is None:
            n, d = X.shape
            s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
            D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
            D = D1 + repmat(s1.T, n, 1)
            np.fill_diagonal(D, 0)
            D = np.sqrt(np.maximum(D, 0))
        else:
            n, d = X.shape
            m, _ = Z.shape
            s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
            s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
            D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
            D = D1 + repmat(s2, n, 1)
            D = np.sqrt(np.maximum(D, 0))
        return D
    
    
    if (kerneltype == "linear"):
        K = np.matmul(X, np.transpose(Z))
    
    elif (kerneltype == "rbf"):
        K = np.exp(-1*kpar*(l2distance(X,Z)**2))
    #for poly
    else:
        K = (np.dot(X, np.transpose(Z)) + 1)**kpar
        

    
    return K


# The following code snippet plots an image of the kernel matrix for the data points in the spiral set. 

get_ipython().run_line_magic('matplotlib', 'inline')
xTr,yTr,xTe,yTe=spiraldata()
K=computeK("rbf",xTr,xTr,kpar=0.05)
# plot an image of the kernel matrix
plt.pcolormesh(K, cmap='jet')
plt.show()
get_ipython().run_line_magic('matplotlib', 'notebook')


import cvxpy as cp
def dualqp(K,yTr,C):
    """
    function alpha = dualqp(K,yTr,C)
    constructs the SVM dual formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        K     | the (nxn) kernel matrix
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
    
    Output:
        alpha | the calculated solution vector (nx1)
    """
    y = yTr.flatten()
    N, _ = K.shape
    alpha = Variable(N)
    
  
    
    alphay = multiply(alpha,y)

 
    constraints = [alpha >=0, alpha <= C, sum(alphay) == 0]
    objective = Minimize(sum(alpha) * (-1) - (0.5) * (quad_form(alphay,K)) * (-1)) 
    prob = Problem(objective, constraints)
    prob.solve()
    
    return np.array(alpha.value).flatten()


C = 10
lmbda = 0.25
ktype = "rbf"
xTr,yTr,xTe,yTe=spiraldata()
# compute kernel (make sure it is PSD)
K = computeK(ktype,xTr,xTr)

alpha=dualqp(K,yTr,C)

# Seeking to classify new test points. 
# Obtaining the bias and the value of the alphas to achieve the same
# b=recoverBias(K,yTr,alphas,C)

def recoverBias(K,yTr,alpha,C):
    """
    function bias=recoverBias(K,yTr,alpha,C);
    Solves for the hyperplane bias term, which is uniquely specified by the 
    support vectors with alpha values 0<alpha<C
    
    INPUT:
    K : nxn kernel matrix
    yTr : nx1 input labels
    alpha  : nx1 vector of alpha values
    C : regularization constant
    
    Output:
    bias : the scalar hyperplane bias of the kernel SVM specified by alphas
    """
    

    value = np.argmin( np.abs(alpha - C/2))
#     print ('value', value)
     
    a = np.multiply(alpha,yTr)
#     b = computeK(ktype, xTr, xTr)[value, :]
    b = K[value]
    bias =  (1/yTr[value] - np.dot(b , a))
#     print ('bias', bias)

    
    return bias


get_ipython().run_line_magic('matplotlib', 'inline')
xTr,yTr=genrandomdata(b=0.5)
C=10
K=computeK("linear",xTr,xTr)
alpha = dualqp(K,yTr,C)
ba=recoverBias(K,yTr,alpha,C)
wa = (alpha * yTr).dot(xTr)
fun = lambda x: x.dot(wa) + ba
visclassifier.visclassifier(fun, xTr, yTr, w=wa, b=ba)

#     Implementing the function 
#     svmclassify=dualSVM(xTr,yTr,C,ktype,kpar);

def dualSVM(xTr,yTr,C,ktype,lmbda):
    """
    function classifier = dualSVM(xTr,yTr,C,ktype,lmbda);
    Constructs the SVM dual formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
        ktype | the type of kernelization: 'rbf','polynomial','linear'
        lmbda | the kernel parameter - degree for poly, inverse width for rbf
    
    Output:
        svmclassify | usage: predictions=svmclassify(xTe);
    """

    
    def svmcfunc(xTe):
        K = computeK(ktype, xTr, xTr, lmbda)
        a = dualqp(K, yTr, C)
        b = recoverBias(K,yTr,a,C)
        return np.dot(np.multiply(a,yTr), computeK(ktype, xTr, xTe, lmbda)) + b
        
    svmclassify = lambda xTe: svmcfunc(xTe)
    
    return svmclassify


xTr,yTr,xTe,yTe=spiraldata()
C=10.0
sigma=0.25
ktype="rbf"
svmclassify=dualSVM(xTr,yTr,C,ktype,sigma)

visclassifier.visclassifier(svmclassify,xTr,yTr)

# compute training and testing error
predsTr=svmclassify(xTr)
trainingerr=np.mean(np.sign(predsTr)!=yTr)
print("Training error: %2.4f" % trainingerr)

predsTe=svmclassify(xTe)
testingerr=np.mean(np.sign(predsTe)!=yTe)
print("Testing error: %2.4f" % testingerr)




def cross_validation(xTr,yTr,xValid,yValid,ktype,CList,lmbdaList):
    """
    function bestC,bestLmbda,ErrorMatrix = cross_validation(xTr,yTr,xValid,yValid,ktype,CList,lmbdaList);
    Use the parameter search to find the optimal parameter,
    Individual models are trained on (xTr,yTr) while validated on (xValid,yValid)
    
    Input:
        xTr      | training data (nxd)
        yTr      | training labels (nx1)
        xValid   | training data (mxd)
        yValid   | training labels (mx1)
        ktype    | the type of kernelization: 'rbf','polynomial','linear'
        CList    | The list of values to try for the SVM regularization parameter C (ax1)
        lmbdaList| The list of values to try for the kernel parameter lmbda- degree for poly, inverse width for rbf (bx1)
    
    Output:
        bestC      | the best C parameter
        bestLmbda  | the best Lmbda parameter
        ErrorMatrix| the test error rate for each given C and Lmbda when trained on (xTr,yTr) and tested on (xValid,yValid),(axb)
    """
    ErrorMatrix=np.zeros((len(CList),len(lmbdaList)))
    bestC,bestLmbda = 0.,0.
    bestTestingerr = 100
    
    counter1 = 0;
    counter2 = 0;
    for C in CList:
        for lmbda in lmbdaList:
            svmclassify = dualSVM(xTr, yTr, C, ktype, lmbda)
            predsTe=svmclassify(xValid)
            testingerr=np.mean(np.sign(predsTe)!=yValid)
#             print(counter1, counter2)
            ErrorMatrix[counter1][counter2] = testingerr
            if(testingerr < bestTestingerr):
                bestTestingerr = testingerr
                bestC = C
                bestLmbda = lmbda
            counter2+= 1
        counter2 = 0
        counter1+= 1
            
    return bestC,bestLmbda,ErrorMatrix


xTr,yTr,xValid,yValid=spiraldata(100)
CList=(2.0**np.linspace(-1,5,7))
lmbdaList=(np.linspace(0.1,0.5,5))
bestC,bestLmbda,ErrorMatrix = cross_validation(xTr,yTr,xValid,yValid,'rbf',CList,lmbdaList)
print(CList, lmbdaList)
print(bestC, bestLmbda)
plt.pcolormesh(ErrorMatrix, cmap='jet')
plt.colorbar()
plt.xlabel("lmbda_idx")
plt.ylabel("C_idx")
plt.title("Validation error")


