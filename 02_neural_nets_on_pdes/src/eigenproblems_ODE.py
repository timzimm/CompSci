#ODE solution for eigenvalues
import numpy as np
import matplotlib.pyplot as plt 

def eig_ODE(A,init_x,dt,iters=1e2):
    N = len(A)
    x = np.zeros([N,iters])
    x[:,0] = init_x
    for t in range(iters-1):
        xt = x[:,t]
        f_x = (xt.T@xt*A+(1-xt.T@A@xt)*np.eye(N))@xt
        x[:,t+1] = xt+dt*(-xt+f_x)
    return x

def dxdt(x,A):
    N= len(A)
    return -x+(x.T@x*A+(1-x.T@A@x)*np.eye(N))@x

dim = 6
T = 1000
A = np.random.randn(dim,dim)
A = 0.5*(A+A.T)
# A = 1/4 * np.array([[1,2,3,4,5],[2,-1,-2,-3,-4],[3,-2,1,1,1],[4,-3,1,0,0],[5,-4,1,0,0]])
init_x = np.random.randn(dim)
init_x = init_x/np.linalg.norm(init_x)

eigval, eigvect = np.linalg.eig(A)
ODE_solv = eig_ODE(A,init_x,0.01,T)

plt.figure(dpi=200)
for i in range(len(ODE_solv.T)): plt.plot(-ODE_solv[:,i],'r',alpha=i/T)
plt.plot(eigvect[:,np.argmax(eigval)],'g')

#%% Phase diagram of 2d matrix
n_points = 50
A = np.random.randn(2,2)
A = 0.5*(A+A.T)
eigval, eigvect = np.linalg.eig(A)
eig1fact, eig2fact = eigvect[1,0]/eigvect[0,0], eigvect[1,1]/eigvect[0,1]
X, Y = np.meshgrid(np.linspace(-10,10,n_points),np.linspace(-10,10,n_points))
coords = np.vstack((X.flatten(),Y.flatten()))
dX = np.zeros([2,n_points**2])
for i in range(n_points**2):
    dX[:,i] = dxdt(coords[:,i].T,A)

sols = []
solN = 20
rand_init = np.random.choice(np.arange(0,n_points**2),replace=False,size=solN)
for i in range(solN):
    sols.append(eig_ODE(A,coords[:,rand_init[i]],0.001,T))

plt.figure(dpi=200)
plt.quiver(X,Y,dX[0],dX[1])
for i in range(solN): plt.plot(sols[i][0],sols[i][1],'b')
plt.plot(np.linspace(-100,100,n_points),eig1fact*np.linspace(-100,100,n_points),'g')
plt.plot(np.linspace(-100,100,n_points),eig2fact*np.linspace(-100,100,n_points),'r')
plt.xlim(-10,10)
plt.ylim(-10,10)

