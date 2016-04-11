#!/usr/bin/env python
"""

The MIT License (MIT)

Copyright (c) 2015 Francesco Mannella <francesco.mannella@gmail.com> 

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np

from esn import ESN

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class ESN_discrete(object) :
    '''
    Learn trajectoriess to goals.
    Changing the spatial goal also 
    changes the shape of the trajectory.
    '''

    def __init__(self, timesteps=1000, lmbd = 1e-4, input_amplitude=50,  **kargs) :
        '''

        timesteps   int:    number of steps of the desired trajectory
        lmbd        int:    ridge regularization parameter
        '''

        self.timesteps = timesteps 
        self.LMBD = lmbd
        self.input_amplitude = input_amplitude
        
        # Init the esn
        self.res = ESN(stime=self.timesteps,  **kargs) 
        # Init the weights to the ESN with sparse random values.
        # The input to the ESN is a 8-elements vector:
        # [ y0[0], 1-y0[0], y0[1], 1-y0[1], goal[0], 1-goal[0], goal[1], 1-goal[1]  ]
        self.input2res_w = np.random.randn(self.res.N, 8) 
        self.input2res_w *= np.random.rand(self.res.N, 8) < 0.1

        self.readout_w = np.zeros([2, self.res.N])


    def imitate_path(self,  y_des) :
        '''
        Learn through ridge regression the weights 
        to the readout unit to reproduce the 'y_des' 
        trajectories.
        '''
        
        Y = np.array([]).reshape(2,0) 
        P = np.array([]).reshape(self.res.N, 0)
        X = np.array([]).reshape(self.res.N, 0)

        for path in y_des :
          
            # learning start-end points
            y0 = path[0]
            goal = path[-1]

            # desired output
            y = np.vstack((self.interpolate(path[:,0]), self.interpolate(path[:,1]))).T
            y = (y - np.outer(np.ones(len(y)),goal)).T

            # input pattern
            p_init = np.dot(self.input2res_w, 
                [ y0[0], 1-y0[0], y0[1], 1-y0[1], 
                    goal[0], 1-goal[0], goal[1], 1-goal[1]])    
            
            # Record the reservoir activity 
            self.res.reset()
            for t in xrange(self.timesteps) :
                p = p_init*(t==0)
                self.res.step(self.input_amplitude*p)
                self.res.store(t)
                # append to the input time-series
                P = np.hstack([P,p.reshape(self.res.N,1)])    
            # network activity time-series 
            x = self.activations()
            #append activities
            X = np.hstack([X,x])
            # append desired outputs
            Y = np.hstack([Y,y])
        
        # Ridge regression
        w = self.readout_w 
        L = self.LMBD
        # including the bias
        N = self.res.N  
        w += np.dot(np.linalg.inv(np.dot(X,X.T) + L*np.eye(N, N)), np.dot(X,Y.T) ).T

    def activations(self) :
        return self.res.data[self.res.out_lab]*\
                np.outer(np.ones(self.res.N),\
                np.exp(-np.linspace(1,0,self.timesteps)))


    def rollout(self, y0=(0., 0.), goal=(1., 1.) ) :
         
        goal = np.array(goal)
        y0 = np.array(y0)

        # input pattern
        p_init = np.dot(self.input2res_w, 
                [ y0[0], 1-y0[0], y0[1], 1-y0[1], 
                    goal[0], 1-goal[0], goal[1], 1-goal[1]])  

        # Record the reservoir activity 
        self.res.reset()
        for t in xrange(self.timesteps) :
            p = p_init*(t==0)
            self.res.step(self.input_amplitude*p)
            self.res.store(t)
        # network activity time-series 
        x = self.activations()
        y = np.dot(self.readout_w, x).T
        y = (y + np.outer(np.ones(len(y)),goal)).T

        return y 


    def interpolate(self, path) :

        import scipy.interpolate
        x = np.linspace( 0, self.timesteps, len(path) )
        y = np.zeros(self.timesteps )
        y_gen = scipy.interpolate.interp1d(x, path)

        for t in range(self.timesteps):  
            y[t] = y_gen(t)
        return y 



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


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
    return None

TIMESTEPS = 200

def gen_trajectory(start, goal, 
                  ax=-0.5, bx=0.001, cx=.5, 
                  ay=0.5,  by=0.001, cy=.48) :
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
    
    # he angle between model and real
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


r = 1.0
xr = 0.0001
NY = 9
NT = NY-1
angle_range = .8
x = np.linspace(xr,-xr, NY)
a = np.pi*((1.0-angle_range)*.5) +(np.pi*angle_range)*(1 -(x+xr)/(xr*2))
rr = a/np.pi*r
start_Y =[]
goal_Y =[]
for t in range(NY):
    start_Y.append( [x[t], 0] )
    goal_Y.append( [x[t] + rr[t]*np.cos(a[t]), rr[t]*np.sin(a[t])] )

Y = []
for k in range(NY):
    x,y,t = gen_trajectory(start_Y[k], goal_Y[k])
    Y.append(np.vstack((x,y)).T )


dx = xr/float(NT)
x = np.linspace(xr-dx,-xr+dx, NT)
a = np.pi*((1.0-angle_range)*.5) +(np.pi*angle_range)*(1 -(x+xr)/(xr*2))
rr = a/np.pi*r
start_T =[]
goal_T =[]
for t in range(NT):
    start_T.append( [x[t], 0] )
    goal_T.append( [x[t] + rr[t]*np.cos(a[t]), rr[t]*np.sin(a[t])] )

T = []
for k in range(NT):
    x,y,t = gen_trajectory(start_T[k], goal_T[k])
    T.append(np.vstack((x,y)).T )
    
XLIM = (-xr-rr[-1]*1.2,xr+rr[1]*1.3)
YLIM = (-rr[-1]*.5,rr[-1])

try :
    import matplotlib
    import matplotlib.pyplot as plt

    cc=matplotlib.colors.LinearSegmentedColormap.from_list(
            'my_cm',[[0,0,1],[0,0,0],[0,.2,0],[0,0,0],[1,0,0]])
    
    def plot_t(x,y,t, xlim_=None, ylim_=None, lwidth = None) :
        
        start = (x[0],y[0])
        goal = (x[-1],y[-1])
        
        if xlim_ == None: xlim_=[-1, 4]
        if ylim_ == None: ylim_=[-1, 4]
        if lwidth == None: lwidth=3
            
        plt.subplot(224)
        plt.plot(x,t)
        plt.xlim(xlim_)
        
        plt.subplot(221)
        plt.plot(t,y)
        plt.ylim(ylim_)
        
        plt.subplot(222)
        for i in xrange(1,len(t)):
            q = (i/float(len(t)))*255.0
            plt.plot(x[(i-1):(i+1)],y[(i-1):(i+1)],lw=lwidth, color=cc(int(q)))
        plt.plot(*start,marker='o', ms=lwidth*2, color='blue')
        plt.plot(*goal,marker='o', ms=lwidth*2, color='red')
        plt.xlim(xlim_)
        plt.ylim(ylim_)
    
    def plot_e(start, goal) :
        x,y,t = gen_trajectory(start, goal)
        plot_t(x,y,t)
     
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------




except :
    print "no graphic device"

if __name__ == "__main__" :

    sim = ESN_discrete()
    
    x = np.linspace(0,1,1000)
    y1 = np.vstack((x,np.sin(x))).T 

    sim.imitate_path([y1])
    trajectory = sim.rollout(y0=(0., 0.), goal=(1., 1.))
    
