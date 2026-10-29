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


def event(t, z):
    # return zero when the active mode's guard is reached
    # solve_ivp then stops, so we can change the discrete mode
    if mode == 'Cooling':
        return z[6] - T_safe    # temperature crosses safe threshold downwar
    if mode == 'Heating':
        return z[6] - T_ready   # temperature crosses printing threshold upward

    elapsed = t - entered - duration
    x, y, vx, vy = position(z)

    # min(...)becomes nonnegative only when ALL conditions are satisfied
    # elapsed time, position proximity and small speed
    if mode == 'MoveV':
        return min(elapsed, position_tol - abs(y), speed_tol - abs(vy))
    if mode == 'MoveH':
        return min(elapsed, position_tol - abs(x), speed_tol - abs(vx))
    if mode == 'Return':
        return min(elapsed, position_tol - np.hypot(x - saved_x, y - saved_y),
                   speed_tol - np.hypot(vx, vy))
    return elapsed              # timed guards for Printing, Wiping and Loading


event.terminal = True           # stop integration at a guard to update the mode

# initial head position, zero motor speed/current and preheated nozzle at equilibrium
x0, y0 = 0.035, 0.020
z = np.array([(x0 + y0)/rm, (x0 - y0)/rm, 0, 0, 0, 0,
              T_ready, u_hold/Rn, u_hold])

# starting mode, a short illustrative print movement before the request
mode, entered, duration = 'Printing', 0, 1.5
start, target = (x0, y0), (x0 + 0.012, y0)
requested = False              # only one replacement request in this simulation
saved_x, saved_y = x0, y0
position_tol, speed_tol = 0.0005, 0.003  # numerical arrival conditions
events = [(0, 'Printing')]
times, xs, ys, temps = [], [], [], []
t, t_end = 0, 20

while t < t_end:
    # thermal cooling crosses downward and all other active events cross upward
    event.direction = -1 if mode == 'Cooling' else 1

    #after printing resumes, continue to t_end without triggering another request
    active_event = None if (mode == 'Printing' and requested) else event
    sol = solve_ivp(sys, (t, t_end), z, events=active_event)

    #keep the ODE solution for plots, without duplicating states at transitions
    times.extend(sol.t[1:] if times else sol.t)
    for state in (sol.y.T[1:] if xs else sol.y.T):
        x, y, _, _ = position(state)
        xs.append(x)
        ys.append(y)
        temps.append(state[6])

    #continue from the last physical state (no jump of physical variables)
    t, z = sol.t[-1], sol.y[:, -1]
    if sol.status != 1:
        break                   #simulation horizon reached, no more event

    x, y, _, _ = position(z)

    #discrete transitions reproduce the filament-replacement FSM of Q2.1
    if mode == 'Printing':
        saved_x, saved_y = x, y
        start, target = (x, y), (x, 0)
        mode, duration, name = 'MoveV', 1, 'Replacement requested'
        requested = True
    elif mode == 'MoveV':
        start, target = (x, y), (0, 0)
        mode, duration, name = 'MoveH', 1, 'Bottom reached'
    elif mode == 'MoveH':
        mode, duration, name = 'Wiping', 0.7, 'Parked / wiping'
    elif mode == 'Wiping':
        mode, name = 'Cooling', 'Wiping finished'
    elif mode == 'Cooling':
        mode, duration, name = 'Loading', 0.7, 'Nozzle cool'
    elif mode == 'Loading':
        mode, name = 'Heating', 'Filament loaded'
    elif mode == 'Heating':
        start, target = (x, y), (saved_x, saved_y)
        mode, duration, name = 'Return', 1, 'Nozzle ready'
    else:                       # Return -> Printing.
        start, target = (x, y), (x + 0.012, y)
        mode, duration, name = 'Printing', 1.5, 'Printing resumed'

    events.append((t, name))
    entered = t                 #restart the clock for the new operation

#prints the detected event times in chronological order
for te, name in events:
    print(f'{te:.2f} s  {name}')

#plot 1: path of the head and the two important positions
plt.figure()
plt.plot(np.array(xs)*1000, np.array(ys)*1000, label='Print head')
plt.plot(0, 0, 'o', label='Wiping location')
plt.plot(saved_x*1000, saved_y*1000, 'x', label='Saved position')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.axis('equal')
plt.legend()
plt.savefig('Q3_1_motion.png')

#plot 2: nozzle temperature, with the two switching thresholds
plt.figure()
plt.plot(times, temps, label='Nozzle temperature')
plt.axhline(T_ready, linestyle='--', label='Printing temperature')
plt.axhline(T_safe, linestyle=':', label='Safe temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.savefig('Q3_1_temperature.png')
plt.show()
