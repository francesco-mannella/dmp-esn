import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

directory = "trajectories"
import os
if not os.path.exists(directory):
    os.makedirs(directory)

stime = 500
t = np.linspace(0.,1.,stime+2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cc = matplotlib.colors.LinearSegmentedColormap.from_list(
    'my_cm',[[1,0,0],[0,0,1]])

params = []
trajectories = []
tt_count = 0 
tl_count = 0

dt = t[1] - t[0]

a_n = 10
b_n =  10
a_s = np.linspace(0.0, 1.0 ,a_n)
b_s = np.linspace(0.0, 1.0 ,b_n)

for a in range(a_n):
    for b in range(b_n):
            A = a_s[a]
            B = b_s[b]

            x = t 
            y = np.ones(t.shape)*A 
            
            z = t+ 0.25*np.exp(-(1./(.0015 + A*0.01))*(t-((B*.3)+.35))**2)
            trajectories.append(z)
            params.append((A, B))

            c = B 
            lw = 2 - 2*(A*0.1-2.0)/6.0  
            plt.plot(x, y, z, color=cc(c))
            zd = (z[1:] -z[:-1])/dt
            zdd = (zd[1:] -zd[:-1])/dt
            zd = zd[1:]
            z = z[2:]
            x = x[2:]
            misc = np.tile((A, B), [len(x),1]).T
            
            if a%2==0 and b%2==0:
                np.savetxt("trajectories/tt_{}".format(tl_count), 
                        np.vstack((x,z,zd,zdd,misc)).T )
                tl_count += 1
            else:
                np.savetxt("trajectories/tl_{}".format(tt_count), 
                        np.vstack((x,z,zd,zdd,misc)).T )
                tt_count += 1

             
plt.show()





