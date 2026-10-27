# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

Tf = 10  # Final simulation time
h = 1  # Sampling time for controller
U = 2  # Control input absolute values
yo = [0.2, 1]  # I.C. (note second entry is initial input, third is time),

# Define differential equation \dot{x}= sat(x^2, 1) + u; \dot{u}=0
def fun(t, u):
    du = [np.min([np.square(u[0]), 1]) + u[1], 0]
    return du


u = yo

ts = []
ys = []
te = []
ye = []
t = 0
tend = 10
c = 1
s = 1000

while t < tend:
    tf = t + h/s
    sol = solve_ivp(fun, (t, tf), u, dense_output=True, max_step=0.05)
    # Store last chuck of simulation
    ts.append(sol.t)
    ys.append(sol.y)
    t = sol.t[-1]
    u = sol.y[:, -1]
    c = c + 1

    # Controller u=-sat(x,U)
    if (c % s) == 0:
        if u[0] >= 1:
            u[1] = -U
        elif u[0] <= -1:
            u[1] = U
        else:
            u[1] = 0


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# We have to stitch together the separate simulation results for plotting
ax.plot(np.concatenate(ts), np.concatenate(ys, axis=1).T)
myleg = plt.legend(['x', 'u'])
fig.show()
