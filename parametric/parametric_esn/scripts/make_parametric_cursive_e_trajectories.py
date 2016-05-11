import numpy as np



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

TIMESTEPS = 200
#PLOT_ONLINE = False 
PLOT_ONLINE = True
ORIGIN = False
#ORIGIN = True

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
    x         float: variance of the x gaussian
    cx         float: mean of the x gaussian
    ay         float: amplitude of the y gaussian
    by         float: variance of the y gaussian
    cy         float: mean of the y gaussian
    '''
 
    # ass = Falure that these are np.arrays
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

import matplotlib.pyplot as plt

def trajectories(cursive_es, angles):
    ts = None
    Y = []
    r1 = -.2
    r2 = .2
    L = []
    A = []
    S = []
    E = []
    for a in  angles: 
        for p in cursive_es:
            start = np.array([r1*np.cos(a), r1*np.sin(a)]) + .5
            end = np.array([r2*np.cos(a), r2*np.sin(a)]) +.5
            A.append( a )
            L.append( p )
            S.append( start )
            E.append( end )
            x,y,t = gen_trajectory(start, end, bx=p, by=p)
            if ts is None: ts = t
            Y.append(np.vstack([x,y]))
    
    return Y, L, A, S, E, ts

def origin_trajectories(cursive_es, angles):
    ts = None
    Y = []
    r1 = 0
    r2 = .4
    L = []
    A = []
    S = []
    E = []
    for a in  angles: 
        for p in cursive_es:
            start = np.array([r1*np.cos(a), r1*np.sin(a)]) + .5
            end = np.array([r2*np.cos(a), r2*np.sin(a)]) +.5
            A.append( a )
            L.append( p )
            S.append( start )
            E.append( end )
            x,y,t = gen_trajectory(start, end, bx=p, by=p)
            if ts is None: ts = t
            Y.append(np.vstack([x,y]))
    
    return Y, L, A, S, E, ts

def plot_trajectories(Y, L, A, ax, color):
    
    for y,i in zip(Y,range(len(Y))):
        y[0,:] = y[0,:] #+ A[i]*3.0 
        y[1,:] = y[1,:] #+ L[i]*3.0
        ax.plot(*y, color=color, lw=.5)
    ax.set_xlim([0,4]) 
    ax.set_ylim([0,4]) 

NY = 9
NT = 9-1
angle_gap = .5
P1 = np.linspace(0.0005, 0.005, NY)
A1 = np.linspace(np.pi*.25, np.pi*(0.25 + angle_gap), NY)
dp2 = (P1[1]-P1[0])*0.5
da2 = (A1[1]-A1[0])*0.5
P2 = np.linspace(0.0005 + dp2, 0.005 - dp2, NT)
A2 = np.linspace(np.pi*.25 + da2, np.pi*(0.25 + angle_gap) - da2, NT)


if not ORIGIN :
    Y, LY, AY, SY, EY, tsY = trajectories(P1,A1)
    T, LT, AT, ST, ET, tsT = trajectories(P2,A2)
else:
    Y, LY, AY, SY, EY, tsY = origin_trajectories(P1,A1)
    T, LT, AT, ST, ET, tsT = origin_trajectories(P2,A2)
    
mmax = np.hstack((AY,AT)).max() 
mmin = np.hstack((AY,AT)).min() 
AY = (AY-mmin)/(mmax-mmin) 
AT = (AT-mmin)/(mmax-mmin) 
    
mmax = np.hstack((LY,LT)).max() 
mmin = np.hstack((LY,LT)).min() 
LY = (LY- mmin)/(mmax-mmin)  
LT = (LT- mmin)/(mmax-mmin) 

if PLOT_ONLINE == True:

     fig = plt.figure()
     ax = fig.add_subplot(211,aspect="equal")

     plot_trajectories(Y[:], LY, AY,  ax, color="blue")
     ax = fig.add_subplot(212,aspect="equal")

     plot_trajectories(T[:], LT, AT, ax, color="red")
     plt.tight_layout()
     plt.show()

else:
    
    def save_formatted_trajectories(Y, L, A, S, E, ts, label ):
        dt = ts[1] - ts[0]
        Y_formatted = []
        for y,i in zip(Y,range(len(Y))):
            ycomp = [ts]
            ys = []
            yds = []
            ydds = []
            for d in range(y.shape[0]):
                cy = y[d,:].ravel()
                yd = np.hstack((0,cy[1:]-cy[:-1]))/dt
                ydd = np.hstack((0,yd[1:]-yd[:-1]))/dt
                ys.append(cy)
                yds.append(yd)
                ydds.append(ydd)
            for t in ys: ycomp.append(t)
            for t in yds: ycomp.append(t)
            for t in ydds: ycomp.append(t)
            ycomp.append(np.ones(len(ts))*A[i])
            ycomp.append(np.ones(len(ts))*L[i])
            ycomp.append(np.ones(len(ts))*S[i][0])
            ycomp.append(np.ones(len(ts))*S[i][1])
            ycomp.append(np.ones(len(ts))*E[i][0])
            ycomp.append(np.ones(len(ts))*E[i][1])
            ycomp = np.vstack(ycomp).T
            np.savetxt(directory+"/{}{:02d}".format(label, i), ycomp, fmt="%10.6f")
    


    save_formatted_trajectories(Y, LY, AY, SY, EY, tsY, "tl")
    save_formatted_trajectories(T, LT, AT, ST, ET, tsT, "tt")
