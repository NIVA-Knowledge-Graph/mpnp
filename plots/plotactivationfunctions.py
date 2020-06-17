import numpy as np
import matplotlib.pyplot as plt

class af:
    "Implementing Activation Functions"

    def identity(x):
        ''' It returns what it recieves, f(x)=y '''
        return x

    def binaryStep(x):
        ''' It returns '0' is the input is less then zero otherwise
                it returns one '''
        return np.heaviside(x, 10)

    def sigmoid(x):
        ''' It returns 1/(1+exp(-x)).
            where the values lies between zero and one '''

        return 1 / (1 + np.exp(-x*10))

    def tanh(x):
        ''' It returns the value (1-exp(-2x))/(1+exp(-2x))
            and the value returned will be lies in between -1 to 1.'''

        return np.tanh(x)

    def arcTan(x):
        ''' It returns the value tanInverse(x)
            and the returned valus lies in between -1.570796327 to 1.570796327. '''

        return np.arctan(x)

    def ReLU(x):
        ''' It returns zero if the inputis less than zero
            otherwise it returns the given input. '''
        x1 = []
        for i in x:
            if i < 0:
                x1.append(0)
            else:
                x1.append(i)

        return x1

    def leakyReLU(x):
        ''' If 'x' is the given input, then returns zero if the inputis less than zero
            otherwise it returns 0.01x . '''

        x1 = []
        for i in x:
            if i < 0:
                x1.append(0.01 * i)
            else:
                x1.append(i)
        # print(x,x1)

        return x1

    def softmax(x):
        ''' Compute softmax values for each sets of scores in x. '''
        return np.exp(x) / np.sum(np.exp(x), axis=0)


def plotActivationFunctions():
    actfn=['Linear','Binary','Sigmoid','ReLU', 'Tanh']
    dic={0:'plt.plot(x, af.identity(x), color="k")',1:'plt.plot(x, af.binaryStep(x), color="k")'
         ,2:'plt.plot(x, af.sigmoid(x), color="k")',3:'plt.plot(x, af.ReLU(x), color="k")',
         4:'plt.plot(x, af.tanh(x), color="k")'}
    for i in range(5):
        x = np.linspace(-1, 1)
        exec(dic[i])
        plt.axis('tight')
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.title('Activation Function: '+actfn[i]+' ')
        plt.show()

def plotActivationFunctions2():
    x = np.linspace(-10, 10)
    x2 = np.linspace(-1, 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    #xx = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    #yy = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]
    xx=[-10,-1,-0.0001,0,0.0001,1,10]
    yy=[0,0,0,1,1,1,1]
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    a = 1
    style = "-"
    #plt.plot(xx,yy)
    plt.plot(x, af.identity(x2), style, label="Linear", alpha=a)
    #plt.plot(x, af.binaryStep(x2), style, label="Binary", alpha=a)
    #plt.plot(xx, yy, style, label="Binary", alpha=a)
    #plt.plot(x, af.sigmoid(x), style, label="Sigmoid", alpha=a)
    #plt.plot(x, af.ReLU(x2), style, label="ReLu", alpha=a)
    plt.plot(x, af.tanh(x), style, label="Tanh", alpha=a)
    plt.legend()

    plt.locator_params(axis='y', nbins=1)
    plt.locator_params(axis='x', nbins=5)

    #plt.axis('tight')
    #plt.axis('equal')
    plt.title('Activation Functions')
    plt.show()


plotActivationFunctions()