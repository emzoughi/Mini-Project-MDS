# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

Tf = 10  # Final simulation time
h = 1  # Sampling time for controller
U = 2  # Control input absolute values
yo = [0.2, 1]  # I.C. (note second entry is initial input, third is time),


# Define event function
def event(t, u):
    if t != 0:
        return (np.absolute(u[0]) - 1)*(t % h)   # Detect also when t multiple of h
    else:
        return (np.absolute(u[0]) - 1)


event.terminal = False


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
tf = h
tend = 10
c = 0
while t < tend:
    sol = solve_ivp(fun, (t, tf), u, events=event, dense_output=True, max_step=0.05)
    # Store last chuck of simulation
    ts.append(sol.t)
    ys.append(sol.y)
    # Store event locations
    if len(sol.t_events[0]) > 0:
        te.append(sol.t_events[0])
        ye.append(sol.y_events[0])
    else:
        c = c + 1
    # Save time and state for initialization next round
    t = sol.t[-1]
    u = sol.y[:, -1]
    # if sol.status == 1: # Event was hit, useful if even.terminal=True
    t = tf
    tf = tf + h
    # Controller u=-sat(x,U)
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
ax.plot(np.concatenate(te), np.concatenate(ye, axis=0), 'o', color='tab:brown')
myleg = plt.legend(['x', 'u'])
fig.show()
