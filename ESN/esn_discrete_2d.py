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

if __name__ == "__main__" :

    sim = ESN_discrete()
    
    x = np.linspace(0,1,1000)
    y1 = np.vstack((x,np.sin(x))).T 

    sim.imitate_path([y1])
    trajectory = sim.rollout(y0=(0., 0.), goal=(1., 1.))
    
