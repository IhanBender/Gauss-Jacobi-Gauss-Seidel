# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg  

def teste1():
    return gauss_jacobi(np.matrix([[10,1],[1,8]]),    \
    np.matrix([[23],[26]]),np.matrix([[0],[0]]),0.00001, 100)


def gauss_jacobi(A,b,x0,tol,N):  
    #preliminares  
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double')  

    #test for graphic representation
    m=np.shape(A)[1]  
    if m != 2:
        raise NameError('Numero de enquações diferente de dois.') 

    n=np.shape(A)[0]
    x = np.zeros(n)
    it = 0  

    #grafico
    ax = plt.axes()
    x1Dots = [x0[0,0]]
    x2Dots = [x0[1,0]]
    x1Gap = [0.0, 0.0]
    x2Gap = [0.0, 0.0]
    plt.xlabel('x1')
    plt.ylabel('x2')

    #iteracoes  
    while (it < N):  
        it = it+1  
        #iteracao de Jacobi  
        for i in np.arange(n):  
            x[i] = b[i]  
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,n))):  
                x[i] -= A[i,j]*x0[j]  
            x[i] /= A[i,i]  

        #inserting results for graphic        
        x1Dots.append(x[0])
        x2Dots.append(x[1])

        #checking if the limits must be extended
        if x[0] < x1Gap[0]:
            x1Gap[0] = x[0]
        elif x[0] > x1Gap[1]:
            x1Gap[1] = x[0]

        #same for x2
        if x[1] < x2Gap[0]:
            x2Gap[0] = x[1]
        elif x[1] > x2Gap[1]:
            x2Gap[1] = x[1]

        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):
            #sets graphic data and returns values 
            for i in range(0, len(x1Dots) - 1):
                plt.arrow(x1Dots[i], x2Dots[i],     \
                         x1Dots[i+1] - x1Dots[i],   \
                         x2Dots[i+1] - x2Dots[i],   \
                         head_width=0, head_length=0, width=0.000001,\
                         fc='k', ec='k', length_includes_head = True)

            plt.xlim(x1Gap[0] - 0.5, x1Gap[1] + 0.5)     # set the x limits
            plt.ylim(x2Gap[0] - 0.5, x2Gap[1] + 0.5)     # set the y limits

            gx = np.arange(-15, 15, 0.001)
            plt.plot(gx, (b[0,0] - (A[0,0]*gx)) / A[0,1], gx, (b[1,0] - (A[1,0]*gx)) / A[1,1], x1Dots, x2Dots, 'bo')
            plt.show() 
            return x  
        
        #prepara nova iteracao
        x0 = np.copy(x)

    raise NameError('num. max. de iteracoes excedido.') 


teste1()



