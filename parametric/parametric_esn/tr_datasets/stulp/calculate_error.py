#!/usr/bin/env python

import sys
import os 
import glob
import numpy as np

pathname = os.path.dirname(sys.argv[0])
if pathname:
    os.chdir(pathname)


def nrmse(y, d):
    y = np.array(y)
    d = np.array(d)
    my = y.max() - y.min()

    return np.sqrt( ((y-d)**2).mean() )/my


n_dim = 1
ts = None


# control existence of folders
theoric = "data/trajectories"
results = 'data/results' 

if not os.path.isdir("data") :
    raise IOError("There's no 'data' folder")

if not os.path.isdir(theoric) :
    raise IOError("There's no '"+theoric+"' folder")

if not os.path.isdir(results) :
    raise IOError("There's no '"+results+"' folder")


# get file names
ftl = glob.glob(theoric+"/tl*")    # training desired
ftt = glob.glob(theoric+"/tt*")    # test desired
frtl = glob.glob(results+"/rtl*")    # training reproduced
frtt = glob.glob(results+"/rtt*")    # test reproduced
for f in [ftl, ftt, frtl, frtt]:
    f.sort()

# all training data
tldata =[]
rtldata =[]
for t,r in zip(ftl, frtl):
    dt = np.loadtxt(t)
    if ts is None: ts = dt.shape[0]
    dr = np.loadtxt(r)
    tldata.append(dt[:ts,1:(n_dim+1)])
    rtldata.append(dr[:ts,1:(n_dim+1)])
tldata = np.vstack(tldata)
rtldata = np.vstack(rtldata)

# all test data
ttdata =[]
rttdata =[]
for t,r in zip(ftt, frtt):
    dt = np.loadtxt(t)
    if ts is None: ts = dt.shape[0]
    dr = np.loadtxt(r)
    ttdata.append(dt[:ts,1:(n_dim+1)])
    rttdata.append(dr[:ts,1:(n_dim+1)])
ttdata = np.vstack(ttdata)
rttdata = np.vstack(rttdata)

# NMSE
tle = nrmse(tldata, rtldata)
tte = nrmse(ttdata, rttdata)
e = (tle + tte)/2.0

print
print "Training NMSE: {:11.10f}".format(tle)
print "    Test NMSE: {:11.10f}".format(tte)
print "   Total NMSE: {:11.10f}".format(e)
print 
