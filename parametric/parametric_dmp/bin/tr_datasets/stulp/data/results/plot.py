#!/usr/bin/env python

import glob 
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
pathname = os.path.dirname(sys.argv[0])
if pathname:
    os.chdir(pathname)

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
ax = fig.add_subplot(111)
for d in trains:
    ax.plot(d[:,1], color="blue", lw=2)
for d in tests:
    ax.plot(d[:,1], color="red",  lw=2, ls='-.')
for d in trial_results:
    ax.plot(d[:,1], color="blue",  lw=1)
for d in test_results:
    ax.plot(d[:,1], color="red",  lw=1)
plt.show()
