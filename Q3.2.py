import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# same parameters
rm, m, Jm = 0.01, 0.06, 1e-5
Km, Lm, Rm = 0.07, 0.01, 10
mu_r, mu_lx, mu_ly = 0.003, 0.002, 0.001
Ln, Cn, Rn = 0.01, 0.002, 8
Ct, Rt, Te = 0.6, 9, 300
T_ready = 465
J = Jm + m*rm**2/2
bx = rm**2*(4*mu_r + mu_lx)
by = rm**2*(2*mu_r + mu_ly)
a, c = (bx + by)/2, (bx - by)/2
b = np.array([bx, by])
u_hold = np.sqrt((T_ready - Te)*Rn/Rt)


def voltages(t, displacement):
    # move between two consecutive corners in 'duration' seconds
    # same fifth-degree polynomial as Q3.1 gives zero endpoint speed and current
    if t >= duration:
        return 0, 0
    s = t/duration
    vel = displacement*(30*s**2 - 60*s**3 + 30*s**4)/duration
    acc = displacement*(60*s - 180*s**2 + 120*s**3)/duration**2
    jerk = displacement*(60 - 360*s + 360*s**2)/duration**3

    current = (J*acc + b*vel)/(rm*Km)
    ux, uy = Lm*(J*jerk + b*acc)/(rm*Km) + Rm*current
    return ux + uy, ux - uy


def sys(t, z):
    thl, thr, wl, wr, il, ir, Tn, inn, vn = z

    # the discrete mode selects which of the four square sides is being drawn
    displacement = np.array(corners[mode + 1]) - np.array(corners[mode])
    ul, ur = voltages(t - entered, displacement)

    # motor voltages drive the movement, the nozzle stays at its holding voltage
    return [wl, wr,
            (Km*il - a*wl - c*wr)/J,
            (Km*ir - c*wl - a*wr)/J,
            (ul - Rm*il)/Lm,
            (ur - Rm*ir)/Lm,
            (vn**2/Rn - (Tn - Te)/Rt)/Ct,
            (u_hold - vn)/Ln,
            (inn - vn/Rn)/Cn]


def event(t, z):
    # target of the current side, converted into angles by Q1.1
    tx, ty = corners[mode + 1]
    target_l, target_r = (tx + ty)/rm, (tx - ty)/rm

    # only motor-angle measurements are used to detect the corner
    error = max(abs(z[0] - target_l), abs(z[1] - target_r))
    # both conditions must hold, planned time elapsed AND angle error small
    return min(t - entered - duration, angle_tol - error)


