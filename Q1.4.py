import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parameters
rm, m, Jm = 0.01, 0.06, 1e-5
Km, Lm, Rm = 0.07, 0.01, 10
mu_r, mu_lx, mu_ly = 0.003, 0.002, 0.001

# desired shapes parameters
T = 8
w = 2*np.pi/T
A, B = 0.04, 0.025             #Ellipse semi axes
R = 0.035                     #Cardioid size

#Equivalent inertia and damping from Q1.2 and 1.3
J = Jm + m*rm**2/2
bx = rm**2*(4*mu_r + mu_lx)
by = rm**2*(2*mu_r + mu_ly)


def dc(k, t, n):
    # n-th time derivative of cos(k*w*t)
    # n=0 gives the position term, n=1, 2, 3 give velocity, acceleration and jerk
    return (k*w)**n*np.cos(k*w*t + n*np.pi/2)


def ds(k, t, n):
    # same thing for sib(k*w*t)
    return (k*w)**n*np.sin(k*w*t + n*np.pi/2)


def desired(t, n, shape):
    # return x and y of the chosen shape, differentiated n times
    if shape == 'ellipse':
        return A*dc(1, t, n), B*ds(1, t, n)

    # Cardioide
    x = R*(dc(1, t, n) - dc(2, t, n)/2)
    y = R*(ds(1, t, n) - ds(2, t, n)/2)
    if n == 0:
        x -= R/2                 # Constant term only present for position
    return x, y


def input_voltages(t, shape):
    # desired derivatives needed by the inverse dynamics of Q1.4
    vx, vy = desired(t, 1, shape)
    ax, ay = desired(t, 2, shape)
    jx, jy = desired(t, 3, shape)

    #mechanical equation gives the desired currents in x/y coordinates
    ix = (J*ax + bx*vx)/(rm*Km)
    iy = (J*ay + by*vy)/(rm*Km)

    #electrical equation u = L*di/dt + R*i
    ux = Lm*(J*jx + bx*ax)/(rm*Km) + Rm*ix
    uy = Lm*(J*jy + by*ay)/(rm*Km) + Rm*iy

    # convert to actual left/right motor voltages, return currents for initialisation
    return ux + uy, ux - uy, ix, iy


def run(shape):
    # start exactly on the desired curve, with matching velocity and current
    x0, y0 = desired(0, 0, shape)
    vx0, vy0 = desired(0, 1, shape)
    _, _, ix0, iy0 = input_voltages(0, shape)
    z0 = [x0, vx0, ix0, y0, vy0, iy0]

    def model(t, z):
        # independent three-state systems from Q1.3
        x, vx, ix, y, vy, iy = z
        ul, ur, _, _ = input_voltages(t, shape)
        ux, uy = (ul + ur)/2, (ul - ur)/2

        # returns [dx/dt, dvx/dt, dix/dt, dy/dt, dvy/dt, diy/dt]
        return [vx, (rm*Km*ix - bx*vx)/J, (ux - Rm*ix)/Lm,
                vy, (rm*Km*iy - by*vy)/J, (uy - Rm*iy)/Lm]

    # integrate the physical model under the calculated motor voltages
    time = np.linspace(0, T, 801)
    sol = solve_ivp(model, (0, T), z0, t_eval=time)
    x_ref, y_ref = desired(time, 0, shape)

    # comparison between simulated head motion and the prescribed shape
    plt.figure()
    plt.plot(sol.y[0]*1000, sol.y[3]*1000, label='Simulated trajectory')
    plt.plot(x_ref*1000, y_ref*1000, '--', label='Desired trajectory')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.legend()
    plt.savefig(f'Q1_4_{shape}.png')


run('ellipse')
run('cardioid')
plt.show()
