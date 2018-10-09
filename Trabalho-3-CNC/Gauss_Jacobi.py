import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg  
 
# jacobi(np.array([[10, 1], [1, 8]]), np.array([[23], [26]]), np.array([0],[0]), 0.0000001, 1000)

def jacobi(A,b,x0,tol,N):  
    #preliminares  
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double')  
 
    n=np.shape(A)[0]  
    x = np.zeros(n)  
    it = 0  

    # Graph
    ax = plt.axes()
    xDots = []
    yDots = []
    xGap = [0, 0]
    yGap = [0, 0]
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
        
        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol): 
            plt.plot(x, (-3 -3*x) / 5.0, x, 2 - 3*x, xDots, yDots, 'bo'
            plt.show() 
            return x  
        
        #prepara nova iteracao
        x0 = np.copy(x)
        #carrega dados para o grafico
        xDots.append(x0[0])
        yDots.append(x0[1])

    raise NameError('num. max. de iteracoes excedido.') 



xList = [1,2,3,4]
yList = [1,4,9,16]

for i in range(0, len(xList) - 1):
    ax.arrow(xList[i], yList[i], \
    xList[i+1]-xList[i], yList[i+1]-yList[i], \
    head_width=0.1, head_length=0.1, fc='k', ec='k', length_includes_head = True)

#ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k', length_includes_head = True)

x = np.arange(0, 17)


