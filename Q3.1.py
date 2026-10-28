import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# same parameters as Q1.4
rm, m, Jm = 0.01, 0.06, 1e-5
Km, Lm, Rm = 0.07, 0.01, 10
mu_r, mu_lx, mu_ly = 0.003, 0.002, 0.001

# nozzle model from Q1.5
Ln, Cn, Rn = 0.01, 0.002, 8
Ct, Rt, Te = 0.6, 9, 300
T_ready, T_safe = 465, 335

#parameters from Q1.2 and 1.3
J = Jm + m*rm**2/2
bx = rm**2*(4*mu_r + mu_lx)
by = rm**2*(2*mu_r + mu_ly)
a, c = (bx + by)/2, (bx - by)/2
b = np.array([bx, by])

#constant nozzle voltage that maintains T_ready at thermal equilibrium
u_hold = np.sqrt((T_ready - Te)*Rn/Rt)


def position(z):
    #kinematics from Q1.1, angles and angular speeds x, y, vx, vy
    thl, thr, wl, wr = z[:4]
    return rm*(thl + thr)/2, rm*(thl - thr)/2, rm*(wl + wr)/2, rm*(wl - wr)/2


def voltages(t, start, target, entered, duration):
    #plan a movement between two points in 'duration' seconds
    # the fifth-degree polynomial starts and ends with zero speed and zero acceleration
    if t - entered >= duration:
        return 0, 0              # the planned movement has finished

    s = (t - entered)/duration  #normalized time between 0 and 1
    d = np.array(target) - np.array(start)
    vel = d*(30*s**2 - 60*s**3 + 30*s**4)/duration
    acc = d*(60*s - 180*s**2 + 120*s**3)/duration**2
    jerk = d*(60 - 360*s + 360*s**2)/duration**3

    #inverse dynamics from Q1.4
    current = (J*acc + b*vel)/(rm*Km)
    # return transformed voltages ux,uy (converted to ul,ur in sys())
    return Lm*(J*jerk + b*acc)/(rm*Km) + Rm*current


def sys(t, z):
    #nine continuous equations, six motor states + three nozzle states
    thl, thr, wl, wr, il, ir, Tn, inn, vn = z

    # the active discrete mode selects which inputs are applied
    if mode in ('Printing', 'MoveV', 'MoveH', 'Return'):
        ux, uy = voltages(t, start, target, entered, duration)
    else:
        ux, uy = 0, 0            # head stationary during the replacement steps
    ul, ur = ux + uy, ux - uy   # CoreXY transformed to actual motor voltages

    # heater off while cooling/loading, full voltage while heating, otherwise hold
    un = 0 if mode in ('Cooling', 'Loading') else (18 if mode == 'Heating' else u_hold)

    #state order: theta_l, theta_r, omega_l, omega_r, i_l, i_r, Tn, in, vn
    return [wl, wr,
            (Km*il - a*wl - c*wr)/J,
            (Km*ir - c*wl - a*wr)/J,
            (ul - Rm*il)/Lm,
            (ur - Rm*ir)/Lm,
            (vn**2/Rn - (Tn - Te)/Rt)/Ct,
            (un - vn)/Ln,
            (inn - vn/Rn)/Cn]


