#!/usr/bin/env python

import glob 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys

matplotlib.use("cairo")

pathname = os.path.dirname(sys.argv[0])
if pathname:
    os.chdir(pathname)

coords = None
if len(sys.argv) == 5:
    try :
        coords = [ float(sys.argv[x]) for x in range(1,5) ]
    except ValueError:
        pass

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
        'color': [.6,.6,1],
        'lw': 5,
        'zorder': 2,
        'label': "Training"
        }
repr_train = {    
        'color': [0,0,.3],
        'lw': 1.5,
        'zorder': 3,
        'label': "Training repr."
        }

theo_test = {    
        'color': [1,.6,.6],
        'lw': 5,
        'zorder': 2, 
        'label': "Test"
        }
repr_test = {    
        'color': [.3,0,0],
        'lw': 1.5, 
        'zorder': 3,
        'label': "Test repr"
        }


def common_plot(ax, d, label, color, lw, zorder):
    h, = ax.plot(d[:,1]+d[:,7]*6, d[:,2]+d[:,8]*6, 
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


fig = plt.figure("DMP Stulp", figsize=(8,8))

ax = fig.add_subplot(111, aspect="equal")

plot_trajectories(ax, trains, lall, **theo_train)    
plot_trajectories(ax, train_results, lall, **repr_train)   


plot_trajectories(ax, tests, lall, **theo_test)   
plot_trajectories(ax, test_results, lall, **repr_test)   

print coords
if coords == None:
    ax.set_xlim([-0.5,10.2])
    ax.set_ylim([-0.5,7.2])
else:
    ax.set_xlim([coords[0], coords[1]])
    ax.set_ylim([coords[2], coords[3]])

ax.set_xticks([])
ax.set_yticks([])

ax.legend(handles=lall)

plt.tight_layout()
plt.show()


