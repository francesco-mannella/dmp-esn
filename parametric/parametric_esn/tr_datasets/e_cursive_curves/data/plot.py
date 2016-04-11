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
lalltrain = []
lalltest = []
idcs = np.arange(len(trains))



theo_train = {    
        'color': [1,.6,.6],
        'lw': 5,
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
        'color': [.6,.6,1],
        'lw': 5,
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
    h, = ax.plot(d[:,1]+d[:,7]*8, d[:,2], 
            color=color, lw=lw, zorder=zorder,
            label=label)
    return h

def plot_trajectories(ax, ttype, lall, **kargs):

    idcs = np.arange(len(ttype))
    for d,i in zip(ttype, idcs):
        if i == 0: 
            lplot = common_plot(ax, d, **kargs)
            lall.append(lplot)
        else:
            common_plot(ax, d, **kargs)




fig = plt.figure("DMP Stulp", figsize=(14,5))

ax = fig.add_subplot(211, aspect="equal")

plot_trajectories(ax, trains, lalltrain, **theo_train)    
plot_trajectories(ax, train_results, lalltrain, **repr_train)   

ax.set_xlim([-0.5,11])
ax.set_ylim([-0.1,1.3])

ax.legend(handles=lalltrain)

ax = fig.add_subplot(212, aspect="equal")

plot_trajectories(ax, tests, lalltest, **theo_test)   
plot_trajectories(ax, test_results, lalltest, **repr_test)   


ax.set_xlim([-0.5,11])
ax.set_ylim([-0.1,1.3])

ax.legend(handles=lalltest)

plt.tight_layout()
plt.show()

