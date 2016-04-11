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
trains = []
for fname in glob.glob("tl*"):
    t = np.loadtxt(fname)
    trains.append(t)

tests = []
for fname in glob.glob("tt*"):
    t = np.loadtxt(fname)
    tests.append(t)

trial_results= []
for fname in glob.glob("rtl*"):
    t = np.loadtxt(fname)
    trial_results.append(t)

test_results= []
for fname in glob.glob("rtt*"):
    t = np.loadtxt(fname)
    test_results.append(t)

fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal")
for d in trains:
    ax.plot(d[:,1] +d[:,7]*6, d[:,2] +d[:,8]*6, color="blue", lw=3, alpha=0.5)
for d in tests: 
    ax.plot(d[:,1] +d[:,7]*6, d[:,2] +d[:,8]*6, color="red",  lw=3, alpha=0.5)
for d in trial_results:
    ax.plot(d[:,1] +d[:,7]*6, d[:,2] +d[:,8]*6, color=[0,0,.5],  lw=2)
for d in test_results:
    ax.plot(d[:,1] +d[:,7]*6, d[:,2] +d[:,8]*6, color=[.5,0,0],  lw=2)
plt.show()

