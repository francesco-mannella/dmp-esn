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

try :

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    matplotlib.use('cairo')
    font = {'family' : 'normal','size'   : 22}
    matplotlib.rc('font', **font)
    cc=matplotlib.colors.LinearSegmentedColormap.from_list(
            'my_cm',[[0,0,1],[0,0,0],[0,.2,0],[0,0,0],[1,0,0]])
    
    class PlotT:

        def __init__(self, margin=0):
            plt.figure(figsize=(10,10))
            
            self.gs = gridspec.GridSpec(10,10)
            self.margin=0
            self.ax1 = plt.subplot(self.gs[:self.margin,self.margin:])
            self.ax2 = plt.subplot(self.gs[self.margin:,:self.margin])
            self.ax3 = plt.subplot(self.gs[self.margin:,self.margin:])
            


        def __call__(self, x,y,t, xlim_=None, ylim_=None, lwidth = None) :
        
            start = (x[0],y[0])
            goal = (x[-1],y[-1])
            
            if xlim_ == None: xlim_=[-1, 4]
            if ylim_ == None: ylim_=[-1, 4]
            if lwidth == None: lwidth=3
                
            self.ax1.plot(x,t,lw=lwidth)
            self.ax1.yaxis.set_label_position("right")
            self.ax1.yaxis.set_ticks_position("right")  
            self.ax1.set_xlim(xlim_)
            self.ax1.set_ylim(-.1,1.1)
            self.ax1.set_ylabel("t")
            self.ax1.set_xticks([])
            
            self.ax2.plot(t,y,lw=lwidth)
            self.ax2.set_xlim(-.1,1.1)
            self.ax2.set_ylim(ylim_)
            self.ax2.set_xlabel("t")
            self.ax2.set_yticks([])

            for i in xrange(1,len(t)):
                q = (i/float(len(t)))*255.0
                self.ax3.plot(x[(i-1):(i+1)],y[(i-1):(i+1)],lw=lwidth, color=cc(int(q)))
            self.ax3.plot(*start,marker='o', ms=lwidth*2, color='blue')
            self.ax3.plot(*goal,marker='o', ms=lwidth*2, color='red')
            if self.margin > 0 :
                self.ax3.yaxis.set_label_position("right")
                self.ax3.yaxis.set_ticks_position("right")
            self.ax3.set_xlim(xlim_)
            self.ax3.set_ylim(ylim_)
            self.ax3.set_xlabel("x")
            self.ax3.set_ylabel("y")
             


    def plot_e(starts=[[0,0]], goals=[[1,1]], bs=[0.001], xlim=None, ylim=None) :
        plot_t = PlotT()
        for start, goal, b in zip(starts, goals, bs): 
            print goal 
            x,y,t = gen_trajectory(start, goal, bx=b, by=b )
            plot_t(x,y,t, xlim_=xlim, ylim_=ylim)
     
except :
    print "no graphic device"

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


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


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------



if __name__ == "__main__" :
    
    plot_e(bs=[0.005], xlim=[-.1,1.1],ylim=[-.1,1.1])
    plt.savefig("trajectory_005.svg")
    plot_e(bs=[0.001], xlim=[-.1,1.1],ylim=[-.1,1.1])
    plt.savefig("trajectory_001.svg")
