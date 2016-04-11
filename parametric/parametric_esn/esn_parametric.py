#!/usr/bin/env python
"""

The MIT License (MIT)

Copyright (c) 2016 Francesco Mannella <francesco.mannella@gmail.com> 

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


import os
import sys
import glob

def log_write(str):
    sys.stdout.write(str)
    sys.stdout.flush()

sys.path.append("../../")

import numpy as np
from ESN.esn_discrete_nd import ESN_discrete

import matplotlib.pyplot as  plt

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

ESN_INP_AMPLIFICATION = 0.5
ESN_LMBD = 1e-3
ESN_N = 600 
ESN_START_WINDOW = 20
ESN_DT = 0.001
ESN_TAU = 0.05
ESN_ALPHA = 0.000001
ESN_BETA = 0.999999
ESN_EPSILON = 1.0e-60 

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
    
if __name__ == "__main__" :

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_of_params',
            help="number of parameters",
            action="store", default=1)
    parser.add_argument('-d','--num_of_dims',
            help="number of dimensions",
            action="store", default=1)
    args = parser.parse_args()
    n_params = int(args.num_of_params)
    n_dim = int(args.num_of_dims)
    ESN_N_DIM = n_dim
   
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    
    # *****GET TRAJECTORIES******

    directory = "data/trajectories"
    results = 'data/results' 
    

    if not os.path.isdir(directory):
        raise ValueError(directory+" does not exist")
    if not os.path.isdir(results):
        raise ValueError(results+" does not exist")
  
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------

    # Get  trajectories
    log_write( "\n")
    log_write( "Reading trajectories ...")
    def get_trajectories(label, n_dim, n_params) :

        # get a list of all training trajectory files
        trajectories_names = glob.glob(directory+"/{}*".format(label))   
        trajectories_names.sort()

        # init lists
        trajectories = []
        params = []
    
        ts = None    # timesteps
        timesteps = None    # number of timesteps 
        further_timesteps = None    # number of further timesteps 

        for tr_name in trajectories_names :
        
            tr = np.loadtxt(tr_name)
            # columns are t, y1, yd1, ydd1, y2, yd2, ydd2, ... yM, ydM, yddM,
            #                param1, param2, ... paramN 
        
            # get the timesteps
            if ts is None:
                ts = tr[:,0].ravel()
                timesteps = len(ts)
                further_timesteps = timesteps/2
        
            # we take all ys
            Y = []
            for y_idx in range(n_dim):
                y = tr[:,1+y_idx].ravel()
                y = np.hstack((y, y[-1]*np.ones(further_timesteps) ))
                Y.append(y)
            Y = np.vstack(Y).T
        
            trajectories.append(Y)         
       
            # then we take parameters (all rows are equal, 
            #       we take the first)
            end_tr = 1+n_dim*3
            prms = tr[0,end_tr:(end_tr+n_params)].ravel()
            params.append(prms)

        return trajectories, params, ts, timesteps, further_timesteps

    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------

    trial_trajectories, trial_params, ts1, ltimesteps, further_ltimesteps = \
            get_trajectories("tl", n_dim, n_params)
    
    test_trajectories, test_params, ts2, ttimesteps, further_ttimesteps = \
            get_trajectories("tt", n_dim, n_params)

    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    log_write(" done.\n")
    
    
    # define timiline variables

    dt1 = ts1[1]-ts1[0]
    ts1 = np.hstack((ts1, np.linspace(ts1[-1]+dt1,
        ts1[-1]+dt1*further_ltimesteps, 
        further_ltimesteps))) 
    
    dt2 = ts2[1]-ts2[0]
    ts2 = np.hstack((ts2, np.linspace(ts2[-1]+dt2,
        ts2[-1]+dt2*further_ttimesteps, 
        further_ttimesteps))) 

    ltimesteps = ltimesteps+further_ltimesteps
    ttimesteps = ttimesteps+further_ttimesteps
   

    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------

    # ****INIT ECHO-STATE****
    log_write( "Initializing ESN ...")

    sim = ESN_discrete(
            n_dim=ESN_N_DIM,
            n_params=n_params,
            inp_amplification=ESN_INP_AMPLIFICATION,
            lmbd=ESN_LMBD,
            N=ESN_N, 
            start_window=ESN_START_WINDOW,
            timesteps=ttimesteps,
            dt=ESN_DT,
            tau=ESN_TAU,
            alpha=ESN_ALPHA,
            beta=ESN_BETA, 
            epsilon=ESN_EPSILON )
  
    log_write( " done.\n")

    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
   
    # ****RUN REGRESSION****
    
    log_write( "Running the regression ...")
   
    sim.imitate_path( trial_trajectories, trial_params)
  
    log_write( " done.\n")

    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------

    # Prepare trial and test rollouts
    
    n_trials = len(trial_trajectories)
    n_tests = len(test_trajectories)
   
    if not os.path.exists(results):
        os.makedirs(results)
    res_names = glob.glob(results+"/rtl*") + glob.glob(results+"/rtt*")
    for name in res_names:
        os.remove(name)
 
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------

    # *****REPRODUCE TRAJECTORIES*****
    
    log_write( "Reproducing the trajectories:\n")

    # reproduce training trajectories
    for x in range(n_trials):
 
        log_write( "Reproducing training {:d}/{:d} ...".format(x,n_trials))

        # running the ESN to reproduce the trajectory
        rtl = sim.rollout(
                y0=trial_trajectories[x][0],
                goal=trial_trajectories[x][-1],
                param=trial_params[x] )

        # formatting the trajectory (calculating first-order and 
        # second-order derivatives and other stuff)
        p = np.outer(np.ones(len(rtl)), trial_params[x])
        
        artl = [ts1.reshape(len(ts1),1)]
        ys = []
        yds = []
        ydds = []
        r = []
        for d in range(rtl.shape[1]):
            rtld = np.hstack((0,rtl[1:,d]-rtl[:-1,d]))/dt1
            rtldd = np.hstack((0,rtld[1:]-rtld[:-1]))/dt1
            ys.append(rtl[:,d])
            yds.append(rtld)
            ydds.append(rtldd)
        
        for t in ys: r.append(t)
        for t in yds: r.append(t)
        for t in ydds: r.append(t)
        
        r = np.vstack(r)

        artl.append(r.T)
        artl.append(p)

        artl = np.hstack(artl)
        
        # saving to file
        np.savetxt("{}/rtl{:02d}".format(results, x),
                artl,
                fmt="%#14.6f")

        log_write( " done\n")

    # reproduce test trajectories
    for x in range(n_tests):
      
        log_write( "Reproducing test {:d}/{:d} ...".format(x,n_tests))

        # running the ESN to reproduce the trajectory
        rtt = sim.rollout(
                y0=test_trajectories[x][0],
                goal=test_trajectories[x][-1],
                param=test_params[x] )
        
        # formatting the trajectory (calculating first-order and 
        # second-order derivatives and other stuff)
        p = np.outer(np.ones(len(rtl)), test_params[x])
        
        artt = [ts2.reshape(len(ts2),1)]
        ys = []
        yds = []
        ydds = []
        r = []
        for d in range(rtt.shape[1]):
            rttd = np.hstack((0,rtt[1:,d]-rtt[:-1,d]))/dt2
            rttdd = np.hstack((0,rttd[1:]-rttd[:-1]))/dt2

            ys.append(rtt[:,d])
            yds.append(rttd)
            ydds.append(rttdd)
      
        
        for t in ys: r.append(t)
        for t in yds: r.append(t)
        for t in ydds: r.append(t)
        
        r = np.vstack(r)
 
        artt.append(r.T)
        artt.append(p)

        artt = np.hstack(artt)
    
    
        # saving to file
        np.savetxt("{}/rtt{:02d}".format(results, x),
                artt,
                fmt="%#14.6f")
        log_write( " done\n")


