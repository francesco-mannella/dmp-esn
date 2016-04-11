#!/usr/bin/env python

import glob 
import numpy as np
import matplotlib.pyplot as plt

import os
import sys



pathname = os.path.dirname(sys.argv[0])
if pathname:
    os.chdir(pathname)

n_dim = None

def get_trajectories(pattern):
    trs = []
    names = glob.glob(pattern)
    names.sort()
    for fname in names:
        t = np.loadtxt(fname)
        trs.append(t)
    return trs


trains = get_trajectories("trajectories/tl*")
tests = get_trajectories("trajectories/tt*")
train_results = get_trajectories("results/rtl*")
test_results = get_trajectories("results/rtt*")

ltrain = None
ltest = None
lall = []
idcs = np.arange(len(trains))



theo_train = {    
        'color': [1,.6,.6],
        'lw': 6,
        'zorder': 2,
        'label': "Training"
        }
repr_train = {    
        'color': [.3,0,0],
        'lw': 1.5,
        'zorder': 3,
        'label': "Training repr."
        }

theo_test = {    
        'color': [.5,.5,1],
        'lw': 6,
        'zorder': 2, 
        'label': "Test"
        }
repr_test = {    
        'color': [0,0,.3],
        'lw': 1.5, 
        'zorder': 3,
        'label': "Test repr"
        }

def common_plot(ax, d, label, color, lw, zorder):
    h, = ax.plot(d[:,0], d[:,1], 
            color=color, lw=lw, zorder=zorder,
            label=label)
    return h

def plot_trajectories(ax, ttype, **kargs):

    idcs = np.arange(len(ttype))
    for d,i in zip(ttype, idcs):
        if i == 0: 
            lplot = common_plot(ax, d, **kargs)
            lall.append(lplot)
        else:
            common_plot(ax, d, **kargs)

fig = plt.figure("DMP Stulp", figsize=(10,4))
ax = fig.add_subplot(111, aspect='auto' )


plot_trajectories(ax, trains, **theo_train)    
plot_trajectories(ax, tests, **theo_test)   
plot_trajectories(ax, train_results, **repr_train)   
plot_trajectories(ax, test_results, **repr_test)   


ax.set_xlim([-0.1,0.8])
ax.set_ylim([-0.1,1.1])

ax.legend(handles=lall)

plt.show()

