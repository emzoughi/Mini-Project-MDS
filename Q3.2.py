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


event.terminal = True            # Ends the ODE segment as soon as a corner is reached
event.direction = 1

# five points are used because the fifth repeats the first to close the square
x0, y0, side = 0.04, 0.04, 0.04
corners = [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side),
           (x0, y0 + side), (x0, y0)]

# start from the first corner, with zero motor speed/current and a heated nozzle
z = np.array([(x0 + y0)/rm, (x0 - y0)/rm, 0, 0, 0, 0,
              T_ready, u_hold/Rn, u_hold])
mode, entered, duration = 0, 0, 1.5
angle_tol = 0.0005
switches = []                   #Event times, one for each completed side
times, xs, ys = [], [], []
t = 0

while mode < 4:
    # integrate the current side only until its angle-based guard becomes true
    sol = solve_ivp(sys, (t, 12), z, events=event)

    #save positions reconstructed from angles, skip the duplicated start point
    times.extend(sol.t[1:] if times else sol.t)
    for state in (sol.y.T[1:] if xs else sol.y.T):
        xs.append(rm*(state[0] + state[1])/2)
        ys.append(rm*(state[0] - state[1])/2)

    # keep the continuous state at the transition
    t, z = sol.t[-1], sol.y[:, -1]
    if sol.status != 1:
        break                    #integration ended without another corner event

    switches.append(t)
    mode += 1                    #switch to the next drawing side
    entered = t                 #reset the elapsed time for the new side

print('Corner times:', ', '.join(f'{te:.2f} s' for te in switches))
print('Corners completed:', mode)

#plot 1: compare the square requested and the motion simulated by the motors
path = np.array(corners)
plt.figure()
plt.plot(path[:, 0]*1000, path[:, 1]*1000, '--', label='Desired square')
plt.plot(np.array(xs)*1000, np.array(ys)*1000, label='Simulated path')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.axis('equal')
plt.legend()
plt.savefig('Q3_2_square.png')

#plot 2: x and y reconstructed from the two motor angles over time
plt.figure()
plt.plot(times, np.array(xs)*1000, label='x')
plt.plot(times, np.array(ys)*1000, label='y')
for te in switches:
    plt.axvline(te, linestyle=':')  #switching instants between square sides
plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.savefig('Q3_2_positions.png')
plt.show()
