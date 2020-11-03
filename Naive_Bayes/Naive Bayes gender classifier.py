# Na&iuml;ve Bayes and the SVM</h2>
# to classify the gender of the baby on the basis of their names 

import numpy as np
import sys
from cvxpy import *
from matplotlib import pyplot as plt
# sys.path.insert(0, './p03/')

# get_ipython().run_line_magic('matplotlib', 'inline')


def feature_extraction_letters(baby, B):
    v = np.zeros(B)
    for letter in baby:
        v[ord(letter) - 97] += 1
    return v

def name2features(filename, B=26, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = feature_extraction_letters(babynames[i].lower(), B)
    return X


# It reads every name in the given file and converts it into a 26-dimensional feature vector by mapping each letter to a feature. 
# genFeatures  transforms the names into features and 
# loads them into memory. 

def genFeatures(dimension, name2features, file_girls, file_boys):
    """
    function [x,y]=genFeatures
    
    This function calls the python script "name2features.py" 
    to convert names into feature vectors and loads in the training data. 
    
    name2features: function that extracts features from names
    dimension: dimensionality of the features
    
    Output: 
    x: n feature vectors of dimensionality d [d,n]
    y: n labels (-1 = girl, +1 = boy)
    """
    
    # Load in the data
    Xgirls = name2features(file_girls, B=dimension)
    Xboys = name2features(file_boys, B=dimension)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]


#Calling the following command to load in the features and the labels of all boys and girls names. 


X,Y = genFeatures(26, name2features, "girls.train", "boys.train")
xTe, yTe = genFeatures(26, name2features, "girls.test", "boys.test")


def naivebayesPY(x,y):
    """
    function [pos,neg] = naivebayesPY(x,y);

    Computation of P(Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)

    Output:
    pos: probability p(y=1)
    neg: probability p(y=-1)
    """
    
    y = np.append(y, [1, -1])

    n = len(y)
    positive = (y==1).sum()
    negative = (y==-1).sum()
    pos = positive/n
    neg = negative/n
 
    return pos,neg

pos,neg = naivebayesPY(X,Y)


def naivebayesPXY_mle(x,y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y) -- Maximum Likelihood Estimate
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (1xd)
    negprob: probability vector of p(x|y=-1) (1xd)
    """
    pos_denom = x[y==1].sum()
    neg_denom = x[y==-1].sum()
    posprob = x[y==1].sum(axis = 0)/pos_denom
    negprob = x[y==-1].sum(axis = 0)/neg_denom
    return posprob, negprob

posprob_mle,negprob_mle = naivebayesPXY_mle(X,Y)


# Estimate the conditional probabilities P(X|Y) (Smoothing with Laplace estimate) in 
# naivebayesPXY_smoothing
# Using a multinomial distribution as model. This will return the probability vectors  for all features given a class label.


def naivebayesPXY_smoothing(x,y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y) -- Smoothing with Laplace estimate
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (1xd)
    negprob: probability vector of p(x|y=-1) (1xd)
    """
  
    shape = x.shape
    d = shape[1] if shape[1:] else 1
    pos_denom = x[y==1].sum() + d
    neg_denom = x[y==-1].sum() + d
    posprob = (x[y==1].sum(axis = 0) + 1)/pos_denom
    negprob = (x[y==-1].sum(axis = 0) + 1)/neg_denom
    return posprob, negprob
    

posprob_smoothing,negprob_smoothing = naivebayesPXY_smoothing(X,Y)


def naivebayes(x,y,xtest,naivebayesPXY):
    """
    function logratio = naivebayes(x,y);
    
    Computation of log P(Y|X=x1) using Bayes Rule
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    xtest: input vector of d dimensions (1xd)
    naivebayesPXY: input function for getting conditional probabilities (naivebayesPXY_smoothing OR naivebayesPXY_mle)
    
    Output:
    logratio: log (P(Y = 1|X=xtest)/P(Y=-1|X=xtest))
    """
    pos, neg = naivebayesPY(x, y)
    posprob, negprob = naivebayesPXY(x, y)
    numerator = np.dot(xtest, np.log(posprob)) + np.log(pos)
    denominator = np.dot(xtest, np.log(negprob)) + np.log(neg)
    logratio = numerator - denominator
    return logratio

p_smoothing = naivebayes(X,Y,X[0,:], naivebayesPXY_smoothing)
p_mle = naivebayes(X,Y,X[0,:], naivebayesPXY_mle)


def naivebayesCL(x,y,naivebayesPXY):
    """
    function [w,b]=naivebayesCL(x,y);
    Implementation of a Naive Bayes classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    naivebayesPXY: input function for getting conditional probabilities (naivebayesPXY_smoothing OR naivebayesPXY_mle)

    Output:
    w : weight vector of d dimensions
    b : bias (scalar)
    """
    
    n, d = x.shape
    pos, neg = naivebayesPY(x, y)
    posprob, negprob = naivebayesPXY(x, y)
    w = np.log(posprob) - np.log(negprob)
    b = np.log(pos) - np.log(neg)
    return w, b

w_smoothing,b_smoothing = naivebayesCL(X,Y, naivebayesPXY_smoothing)
w_mle,b_mle = naivebayesCL(X,Y, naivebayesPXY_mle)

# Implimenting classifyLinear that applies a linear weight vector and bias to a set of input vectors and outputs their predictions

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b);
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector of d dimensions
    b : bias (optional)
    
    Output:
    preds: predictions
    """
    
    # print(x.shape)
    # print(np.transpose(w).shape)
    preds = np.matmul(x,np.transpose(w))
    preds = preds + b
    np.place(preds,preds==0,-1)
    preds = np.sign(preds)
    
    return preds

print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 *(classifyLinear(X, w_smoothing, b_smoothing) != Y).mean()))
print('Training error (Maximum Likelihood Estimate): %.2f%%' % (100 *(classifyLinear(X, w_mle, b_mle) != Y).mean()))
print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 *(classifyLinear(xTe, w_smoothing, b_smoothing) != yTe).mean()))
print('Test error (Maximum Likelihood Estimate): %.2f%%' % (100 *(classifyLinear(xTe, w_mle, b_mle) != yTe).mean()))




DIMS = 26
print('Loading data ...')
X,Y = genFeatures(DIMS, name2features, "girls.train", "boys.train")
xTe, yTe = genFeatures(26, name2features, "girls.test", "boys.test")
print('Training classifier (Smoothing with Laplace estimate) ...')
w,b=naivebayesCL(X,Y,naivebayesPXY_smoothing)
train_error = np.mean(classifyLinear(X,w,b) != Y)
test_error = np.mean(classifyLinear(xTe,w,b) != yTe)
print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 * train_error))
print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 * test_error))

yourname = ""
while yourname!="exit":
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname,B=DIMS,LoadFile=False)
    pred = classifyLinear(xtest,w,b)[0]
    if pred > 0:
        print("%s, I am sure you are a nice boy.\n" % yourname)
    else:
        print("%s, I am sure you are a nice girl.\n" % yourname)


# Feature Extraction


def feature_extraction_letters_pairs(name, B=676):
    """
    Feature extration from name for pairs
    name: name of the baby as a string
    
    Output:
    v : a feature vectors of dimension B=676, (B,)
    """
    v = np.zeros(B)
    i = 0
    
    ken = np.zeros((26,26))
    while (i<(len(name)-1)):
            zep = ord(name[i])-97
            zap = ord(name[i+1])-97
            ken[zep][zap] = ken[zep][zap] + 1
            i = i+1
    v = ken.flatten()
    return v
    
def name2features_pairs(filename, B=676, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = feature_extraction_letters_pairs(babynames[i].lower(), B)
    return X


''' result of the Naive Bayes classifier using pairs of letters as features '''
DIMS = 676
print('Loading data ...')
Xp,Yp = genFeatures(676, name2features_pairs, "girls.train", "boys.train")
xTe, yTe = genFeatures(676, name2features_pairs, "girls.test", "boys.test")
print('Training classifier (Smoothing with Laplace estimate) ...')
w,b=naivebayesCL(Xp,Yp,naivebayesPXY_smoothing)
train_error = np.mean(classifyLinear(Xp,w,b) != Yp)
print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 * train_error))
test_error = np.mean(classifyLinear(xTe,w,b) != yTe)
print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 * test_error))

yourname = ""
while yourname!="exit":
    print('Please enter your name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features_pairs(yourname,B=DIMS,LoadFile=False)
    pred = classifyLinear(xtest,w,b)[0]
    if pred > 0:
        print("%s, I am sure you are a nice boy.\n" % yourname)
    else:
        print("%s, I am sure you are a nice girl.\n" % yourname)


#SVM vs Naive Bayes
# We will now explore the performance of soft-margin SVM in comparison to Naive Bayes on the same dataset:

''' Implementation of soft-margin SVM '''

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
    
    w = Variable(d)
    b = Variable(1)
    e = Variable(N)
    objective = sum_squares(w) + C*sum(e)
    constraints = [e >= 0, 
                   multiply(y, xTr*w + b) >= 1-e]
    prob = Problem(Minimize(objective), constraints)
    prob.solve()
    wout = w.value
    bout = b.value
    eout = e.value
    
    fun = lambda x: x.dot(wout) + bout
    return fun, wout, bout


# Comparison of results for SVM vs Naive Bayes using single-letter features


C=20

xTr,yTr = genFeatures(26, name2features, "girls.train", "boys.train")
xTe, yTe = genFeatures(26, name2features, "girls.test", "boys.test")
fun, _, _ = primalSVM(xTr, yTr, C)

svm_err_tr1=np.mean(np.array((np.sign(fun(xTr)))!=yTr).flatten())
print("Training error using SVM: %2.1f%%" % (svm_err_tr1*100))
nb_w,nb_b=naivebayesCL(xTr,yTr,naivebayesPXY_smoothing)
nb_train_error1 = np.mean(classifyLinear(xTr,nb_w,nb_b) != yTr)
print('Training error using Naive Bayes with smoothing: %.2f%%' % (100 * nb_train_error1))

svm_err_te1=np.mean(np.array((np.sign(fun(xTe)))!=yTe).flatten())
print("Test error using SVM: %2.1f%%" % (svm_err_te1*100))
nb_test_error1 = np.mean(classifyLinear(xTe,nb_w,nb_b) != yTe)
print('Test error using Naive Bayes with smoothing: %.2f%%' % (100 * nb_test_error1))


# Comparison of results for SVM vs Naive Bayes using paired-letter features


xTr,yTr = genFeatures(676, name2features_pairs, "girls.train", "boys.train")
xTe, yTe = genFeatures(676, name2features_pairs, "girls.test", "boys.test")
fun, _, _ = primalSVM(xTr, yTr, C)

svm_err_tr2=np.mean(np.array((np.sign(fun(xTr)))!=yTr).flatten())
print("Training error using SVM: %2.1f%%" % (svm_err_tr2*100))
nb_w,nb_b=naivebayesCL(xTr,yTr,naivebayesPXY_smoothing)
nb_train_error2 = np.mean(classifyLinear(xTr,nb_w,nb_b) != yTr)
print('Training error using Naive Bayes with smoothing: %.2f%%' % (100 * nb_train_error2))

svm_err_te2=np.mean(np.array((np.sign(fun(xTe)))!=yTe).flatten())
print("Test error using SVM: %2.1f%%" % (svm_err_te2*100))
nb_test_error2 = np.mean(classifyLinear(xTe,nb_w,nb_b) != yTe)
print('Test error using Naive Bayes with smoothing: %.2f%%' % (100 * nb_test_error2))


# Visual Comparison


nb = [100 * nb_train_error1, 100 * nb_test_error1, 100 * nb_train_error2, 100 * nb_test_error2]
svm = [svm_err_tr1 * 100, 100 * svm_err_te1, svm_err_tr2 * 100, 100 * svm_err_te2]
n_groups = 4

fig, ax = plt.subplots(figsize=(6,4.5), dpi = 100)
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, nb, bar_width,
alpha=opacity,
color='b',
label='NaiveBayes')

rects2 = plt.bar(index + bar_width, svm, bar_width,
alpha=opacity,
color='r',
label='SVM')

plt.xlabel('')
plt.ylabel('Error (% age)')
plt.title('Error Rates Comparison')
plt.xticks(index + bar_width, ('letter-train', 'letter-test', 'pairs-train', 'pairs-test'))
plt.legend()

plt.tight_layout()
plt.show()

