import numpy as np



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

TIMESTEPS = 200
PLOT_ONLINE = True

def get_angle(v1,v2) :
    """
    Calculate the angle between two vectors
    v1  (array):   first vector
    v2  (array):   second vector
    """

    if (np.linalg.norm(v1)*np.linalg.norm(v2)) != 0 :     
        cosangle = np.dot(v2,v1)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        cosangle = np.maximum(-1,np.minimum(1, cosangle))
        angle = np.arccos(cosangle)   
        if np.cross(v2,v1) < 0 :
            angle = 2*np.pi - angle  
        return angle
    return 2*np.pi


def gen_trajectory(start, goal, 
                  ax=-0.3, bx=0.001, cx=.5, 
                  ay=0.3,  by=0.001, cy=.48) :
    '''
    Generate an 'e' trajectory
    
    start    (array): start point
    goal     (array): end point
    ax         float: amplitude of the x gaussian
    bx         float: variance of the x gaussian
    cx         float: mean of the x gaussian
    ay         float: amplitude of the y gaussian
    by         float: variance of the y gaussian
    cy         float: mean of the y gaussian
    '''
    
    # assure that these are np.arrays
    start = np.array(start)
    goal = np.array(goal)
    
    # the end point of the model trajectory
    unit = [1,1]
    # the end point of the real trajectory 
    # translated to the origin
    target = goal - start
    
    # the angle between model and real
    angle = get_angle(unit, target) 
    
    # create model trajectory
    t = np.linspace(0,1,TIMESTEPS)
    xunit = ax*np.exp(-(1./bx)*(t-cx)**2)+t
    yunit = ay*np.exp(-(1./by)*(t-cy)**2)+t
    
    # rotate
    X = xunit*np.cos(angle) + yunit*np.sin(angle)
    Y = -xunit*np.sin(angle) + yunit*np.cos(angle)
    
    # scale and traslate
    scale = np.linalg.norm(target)/np.linalg.norm(unit)
    X = X*scale + start[0]
    Y = Y*scale + start[1]
    
    return X,Y,t




r = .15
xr = 0.1
NY = 9
NT = NY-1

atot = np.linspace(np.pi,0,NY+NT)
xtot = 0.5 + np.linspace(-xr,xr, NY+NT)
rrtot = .2 + np.linspace(0,1,NY+NT)*r


angle_range = 0.9
x = xtot[np.arange(NY)*2]
a =  atot[np.arange(NY)*2]
rr = rrtot[np.arange(NY)*2]

start_Y =[]
goal_Y =[]

for t in range(NY):
    start_Y.append( [x[t], 0.5] )
    goal_Y.append( [0.5+rr[t]*np.cos(a[t]), 0.5+rr[t]*np.sin(a[t])] )

Y = []
for k in range(NY):
    x,y,t = gen_trajectory(start_Y[k], goal_Y[k])
    Y.append(np.vstack((x,y)).T )


dx = xr/float(NT)
da = np.pi/float(NT)
x = xtot[np.arange(NT)*2+1]
a =  atot[np.arange(NT)*2+1]
rr = rrtot[np.arange(NT)*2+1]
start_T =[]
goal_T =[]
for t in range(NT):
    start_T.append( [x[t], 0.5] )
    goal_T.append( [0.5 + rr[t]*np.cos(a[t]), 0.5+rr[t]*np.sin(a[t])] )

T = []
for k in range(NT):
    x,y,t = gen_trajectory(start_T[k], goal_T[k])
    T.append(np.vstack((x,y)).T )


ts = np.linspace(0,1,TIMESTEPS)

import os
import glob
directory = "data/trajectories"
results = 'data/results' 

if not os.path.isdir("data") :
    os.mkdir("data")
    os.mkdir(directory)
    os.mkdir(results)
if not os.path.isdir(directory) :
    os.mkdir(directory)
tr_names = glob.glob(directory+"/tl*")+ glob.glob(directory+"/tt*")
for name in tr_names:
    os.remove(name)

if PLOT_ONLINE:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

dt = ts[1]-ts[0]
params = np.linspace(0,1,len(Y))
for y,i in zip(Y,range(len(Y))):

    ycomp = [ts]
    for d in range(y.shape[1]):
        cy = y[:,d].ravel()
        yd = np.hstack((0,cy[1:]-cy[:-1]))/dt
        ydd = np.hstack((0,yd[1:]-yd[:-1]))/dt
        ycomp.append(cy)
        ycomp.append(yd)
        ycomp.append(ydd)
    
    ycomp.append(params[i]*np.ones(len(cy)))

    ycomp = np.vstack(ycomp).T
    
    if PLOT_ONLINE:   
        ax.plot(ycomp[:,1],ycomp[:,4], color="blue")
    else:
        np.savetxt(directory+"/tl{:02d}".format(i), ycomp,fmt="%10.6f")




params = params[1]/2 + np.linspace(0,1-params[1],len(T))
for y,i in zip(T,range(len(T))):

    ycomp = [ts]
    for d in range(y.shape[1]):
        cy = y[:,d].ravel()
        yd = np.hstack((0,cy[1:]-cy[:-1]))/dt
        ydd = np.hstack((0,yd[1:]-yd[:-1]))/dt
        ycomp.append(cy)
        ycomp.append(yd)
        ycomp.append(ydd)
    
    ycomp.append(params[i]*np.ones(len(cy)))

    ycomp = np.vstack(ycomp).T

    if PLOT_ONLINE:
        ax.plot(ycomp[:,1],ycomp[:,4], color="red")
    else:
        np.savetxt(directory+"/tt{:02d}".format(i), ycomp,fmt="%10.6f")

if PLOT_ONLINE:
    plt.show()
