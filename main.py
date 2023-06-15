import math

import numpy as np
import matplotlib.pyplot as plt

def edo2(i):

    h = round(10**(-i),i)
    n = int(5 / h)
    A = np.zeros((n-1,n-1))
    b = np.zeros(n-1)
    x = [1+h]

    for j in range(0,n-2):
        x+= [round(x[j]+h,i)]

    for i in range(0,n-1):

        A[i][i] = YJ(h,x[i])
        if i<n-2:
            A[i][i+1] = YJm1(h, x[i])
        if i>0:
            A[i][i - 1] = YJp1(h, x[i])

    b[0] = ((1 / (h ** 2)) + (1 / (2 * h * x[0])))


    print(x)

    xv = []
    yv = []
    for i in range(100,600):
        xv+= [i/100]
        yv+= [fun(i/100)]

    res = np.linalg.solve(A,b)
    print(len(res))
    plt.plot(x,res)
    plt.plot(xv,yv)
    plt.show()

    return res

def YJp1(h,x):
    return ((-1/(h**2))+(1/(2*h*x)))


def YJ(h,x):
    return((2/ pow(h,2))+(1/(4*(x**2)))-1)


def YJm1(h,x):
    return ((-1/(h**2)) - (1/(2*h*x)))

def fun(x):
    return math.sin(6-x)/(math.sin(5)*math.sqrt(x))
print(edo2(3))
