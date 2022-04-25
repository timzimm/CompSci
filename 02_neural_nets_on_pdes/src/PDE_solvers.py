import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#%% initialize grid/parameters
lwr_lim, hgr_lim = 0, 1
T = 1
midpoint, length = (hgr_lim-lwr_lim)/2, hgr_lim-lwr_lim
dt,dx = 0.00005, 0.01 #s & cm
print(dt/dx**2)
xsteps, tsteps = int(length/dx), int(T/dt)
x = np.linspace(lwr_lim,hgr_lim,xsteps)
x_ = np.expand_dims(x,1)
time = np.linspace(0,T,tsteps)
t_ =  np.expand_dims(time,1)

init_cond = np.sin(np.pi*x)#np.exp(-2*(x-midpoint)**2)+0.25*np.exp(-2*(x-15)**2)
alpha = dt/dx**2

#%%Analytic solution for the givien initial/boundary conditions
f_xt = np.exp(-(np.pi**2)*t_)*np.sin(np.pi*x_.T)

#%%Explicit scheme heat equation

#u_{:,j+1} = au_{:,j+1}+(1-2a)u_{:,j}+au_{:,j-1} - fdiff expression
V = np.zeros([xsteps,tsteps])
V[:,0] = init_cond

A1 = alpha*np.ones((xsteps-1))
A0 = (1-2*alpha)*np.ones((xsteps))
A = np.diag(A1,1)+np.diag(A0,0)+np.diag(A1,-1)


for t in range(tsteps-1):
    V[:,t+1] = A@V[:,t]
    V[0,t+1] = 0
    V[-1,t+1] = 0

#Spectra + spectral radius
eigv = np.linalg.eig(A)[0]

print('Maximal spectral radius of A is '+str(np.max(np.abs(eigv))))
#%%Plots
# plt.figure(dpi=200)
# plt.plot(f_xt.T,'k',alpha=0.2)
# plt.plot(V,'r',alpha=0.2)
# plt.title('Explicit scheme')
# print('Squared error = '+str(np.sum((f_xt.T-V)**2)))

plt.figure(dpi=200,figsize=(8,2))
plt.pcolormesh(time,x,V)

plt.figure(dpi=200,figsize=(8,2))
plt.pcolormesh(time,x,f_xt.T-V,cmap='coolwarm',norm=colors.CenteredNorm())
plt.colorbar()
plt.title('Error for $\Delta$x='+str(dx)+' & $\Delta$t='+str(dt))
#%%Implicit scheme

#u_{:,j-1}=-au_{i+1,j}+(1-2a)u_{i,j}-au_{i+1,j}
Ap0 = (1+2*alpha)*np.ones((xsteps))

Ap = -np.diag(A1,1)+np.diag(Ap0,0)-np.diag(A1,-1)
Ap_inv = np.linalg.inv(Ap) #Do with LU decomp

Vp = np.zeros([xsteps,tsteps])
Vp[:,0] = init_cond

for t in range(1,tsteps):
    Vp[:,t] = Ap_inv@Vp[:,t-1]
    Vp[0,t] = 0
    Vp[-1,t] = 0

#Spectra + spectral radius
eigv = np.linalg.eig(Ap_inv)[0]

print('Maximal spectral radius of A is '+str(np.max(np.abs(eigv))))
#%%Plots
# plt.figure(dpi=200)
# plt.plot(f_xt.T,'k',alpha=0.2)
# plt.plot(Vp,'b',alpha=0.2)
# plt.title('implicit scheme')
# print('Squared error = '+str(np.sum((f_xt.T-Vp)**2)))

plt.figure(dpi=200,figsize=(8,2))
plt.pcolormesh(time,x,Vp)
plt.title('Implicit scheme for $\Delta$x='+str(dx)+' & $\Delta$t='+str(dt))

# plt.figure()
# plt.plot(np.real(eigv),np.imag(eigv),'.')



plt.figure(dpi=200,figsize=(8,2))
plt.pcolormesh(time,x,f_xt.T-Vp,cmap='coolwarm',norm=colors.CenteredNorm())
plt.colorbar()
plt.title('Error for $\Delta$x='+str(dx)+' & $\Delta$t='+str(dt))
