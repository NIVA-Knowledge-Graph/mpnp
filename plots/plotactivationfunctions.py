import numpy as np
import matplotlib.pyplot as plt


def linear(x):
    return x


def binary(x):
    return np.heaviside(x, 10)


def sigmoid(x):
    return 1 / (1 + np.exp(-x*10))


def tanh(x):
    return np.tanh(x)


def ReLU(x):
    x1 = []
    for i in x:
        if i < 0:
            x1.append(0)
        else:
            x1.append(i)
    return x1


def plotActivationFunctions():
    actfn=['Linear','Binary','Sigmoid','ReLU', 'Tanh']
    dic={0:'plt.plot(x, linear(x), color="k")',1:'plt.plot(x, binary(x), color="k")'
         ,2:'plt.plot(x, sigmoid(x), color="k")',3:'plt.plot(x, ReLU(x), color="k")',
         4:'plt.plot(x, tanh(x), color="k")'}
    for i in range(5):
        x = np.linspace(-1, 1)
        exec(dic[i])
        plt.axis('tight')
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.title('Activation Function: '+actfn[i]+' ')
        plt.show()

plotActivationFunctions()