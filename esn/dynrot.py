#!/usr/bin/python

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import sys

def get_matrix(n=200, Winit=np.array([]),  alpha=0.5, radius=1.0, h=0.1, epsilon=1e-4): 
    '''
    Build a random nxn matrix of weights.

    n (int):            Number of variables
    Winit array(float): Number of variables
    alpha (float):      Proportion of infinitesimal rotation
    radius (float):     Desired spectral radius of the matrix
    h (float):          Decay rate (1/tau) 
    epsilon (float):    Infinitesimal distance from the chaotic boundary 
    '''

    # initialize random matrix
    if Winit.size == 0 :
        # build a matrix based on alpha
        W = rnd.randn(n,n) 
    else:
        W = Winit.copy()

    # decompose
    W1 = (W - W.T)/2.   # rotation
    W2 = (W + W.T)/2.   # expansion/rotation
    
    # recompose
    W = alpha*W1 + (1-alpha)*W2     
  
    # # compute the infinitesimal dynamic matrix of the system dx = -x + W*tanh(x)  
    dW = h*W + (1-h)*np.eye(n,n)         
    # # scale so that the spectral radius is 'radius'
    dW = (1-epsilon)*((radius*dW)*(1.0/np.max(np.abs(np.linalg.eigvals(dW)))))
    # # extract W from dW 
    W = (1.0/h)*(dW - (1-h)*np.eye(n,n))
     
    return W
    
def run_dynamics(W, m=100, n =200,  h=0.1) :
    ''' 
    Run the dynamics of the system and compute PCA on 
    the time-series.
    
    dx = -x + W*tanh(x)
    
    W (nxn matrix):  Matrix of weights of the system
    m (int):         number of timesteps
    n (int):         Number of independent variables
    h (float):       decay rate (1/tau) 

    return:
    X (Mxn):         time series of the n variables 
    T (Mx3):         time series of on the first 3 principal components
    '''


    X = np.zeros([m,n])    # data storage
    x = rnd.randn(n)  
 
    def fun(x) : 
        return np.tanh(x)

    # integrate over time
    for t in range(m) :
        x += h*(-x + np.dot(W,fun(x))) 
        X[t,:] = fun(x)

    #PCA
    B = X - X.mean(0)   # subtract mean
    C = np.dot(B.T,B)/float(n-1)    # covariance matrix
    E, M = np.linalg.eig( np.dot(C.T,C) )     # eigenvalues, eigenvectors
    E = np.real(E)    # throw off 0j imag part
    M = np.real(M[np.argsort(E)[::-1]])    # sort eigenvector (throw off 0j imag part)
    M = M[:,:3]    # get first 3 eigenvectors
    T = np.dot(B,M)   # first 3 principal components

    return X,T


from matplotlib.gridspec import GridSpec 
from mpl_toolkits.mplot3d import Axes3D
def plot_matrix(W, X, T, fig, grid_pos ) :
    '''
    Plot the spectrogram of the weight matrix,
    the timeseries and the trajectory of the first 3 principal components

    W (nxn matrix):             Matrix of weights of the system
    X (Mxn):                    time series of the n variables 
    T (Mx3):                    time series of on the first 3 principal components
    fig (Figure object):
    grid_pos list(GridSpecs):
    '''
   
    stime = X.shape[0]

    # plot the spectrogram of the W matrix 
    ax1 = fig.add_subplot(grid_pos[0])
    EM, _ = np.linalg.eig( W )     
    ax1.scatter(np.real(EM),np.imag(EM))
    radius = np.max(np.abs(EM)) 
    ax1.set_xlim([-radius, radius])
    ax1.set_ylim([-radius, radius])
    ax1.xaxis.set_ticks([-radius,0, radius])
    ax1.yaxis.set_ticks([-radius,0, radius])


    
    # plot the time-series
    ax2 = fig.add_subplot(grid_pos[1])
    ax2.plot(X) 
    ax2.xaxis.set_ticks([0, stime])

   
    # plot the time-series  zoom
    ax3 = fig.add_subplot(grid_pos[2])
    ax3.plot(X[int(stime*0.1):int(stime*0.15),:])  
    ax3.xaxis.set_ticks([0, int(stime*0.05)])
    ax3.xaxis.set_ticklabels([int(stime*0.1), int(stime*0.15)])
    
    # plot the time-series  zoom
    ax4 = fig.add_subplot(grid_pos[3])
    ax4.plot(X[int(stime*0.5):int(stime*0.55),:])  
    ax4.xaxis.set_ticks([0, int(stime*0.05)])
    ax4.xaxis.set_ticklabels([int(stime*0.5), int(stime*0.55)])
    
    # plot the trajectory made by the first 3 principal components
    ax3 = fig.add_subplot(grid_pos[4],projection="3d")
    ax3.plot(T[:,0],T[:,1],T[:,2])
    ax3.scatter(T[0,0],T[0,1],T[0,2],s=60,c ="#ff6666")
    ax3.scatter(T[-1::,0],T[-1::,1],T[-1::,2],c="blue")
    ax3.xaxis.set_ticks([  ])
    ax3.yaxis.set_ticks([  ])
    ax3.zaxis.set_ticks([  ])
  
    
    
    fig.canvas.draw()

def demo() :

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # GRAPHICS 
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(9, 6)
    labels = ["spectrogram", 
            "time-series",
            "zoom1", 
            "zoom2", 
            "PCA\ntrajectory"]

    for k in xrange(1,6):
        ax = fig.add_subplot(gs[0,k])
        ax.text(.2,.5,labels[k-1], size=20)
        ax.set_axis_off()
    # ------------------------------------------------------------
    # ------------------------------------------------------------
  
    count = 1
   
    n = 200    # number of units 
    m = 1000 
    
    # all simulation start from the same random population of weights
    Winit = rnd.randn(n,n)
    
    # iterate alpha parameter - balance betweeen rot and exp/contr  
    for alpha in np.linspace(.1,.9,8) :
         
        # build the weight matrix
        W = get_matrix(n = n, Winit = Winit, alpha = alpha, h = 0.1)
        # run the dynamics
        X,T = run_dynamics(W = W, n = n, m = m, h = 0.1) 
  
        # --------------------------------------------------------
        # --------------------------------------------------------
        # GRAPHICS
        label = "alpha={:4.2f} ".format(alpha )
        ax = fig.add_subplot(gs[count,0])
        ax.text(.05,.5,label, size=20)
        ax.set_axis_off()
        gs_row = [ gs[count, k] for k in xrange(1,6)]
        plot_matrix(W, X, T,fig, gs_row )   
        # --------------------------------------------------------
        # --------------------------------------------------------

        count += 1

    plt.tight_layout()


############################################################################

if __name__ == "__main__" :
    

    plt.ion()
    plt.close("all")
    
    demo()

    raw_input()

