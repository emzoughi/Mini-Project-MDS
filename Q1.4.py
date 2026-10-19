import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

rm=0.01
m =0.06
Jm =1e-5
Km= 0.07
Lm =0.01
Rm =10
mu_r=0.003
mu_lx= 0.002
mu_ly=0.001
T =8
w = 2* np.pi / T
A, B =0.04, 0.025
R= 0.035
J = Jm+ m *rm**2 /2
bx= rm**2 *(4 * mu_r + mu_lx)
by =rm**2 * (2 * mu_r + mu_ly)

#nth time derivative of cos(k*w*t)
def dc(k, t, n):
    return (k * w)**n * np.cos(k * w * t + n * np.pi / 2)

#nth time derivative of sis(k*w*t)
def ds(k, t, n):
    return (k * w)**n *np.sin(k * w * t + n * np.pi/ 2)


#n = 0 position, 1 velocity, 2 acceleration, 3 jerk
def desired(t, n, shape):
    if shape =="ellipse":
        return A*dc(1, t, n), B*ds(1, t, n)

    #Cardioide
    x = R * (dc(1, t, n) - dc(2, t, n)/2)
    y = R * (ds(1, t, n) - ds(2, t, n)/2)
    if n == 0:
        x -= R/2
    return x, y

def input_voltages(t, shape):
    vx, vy = desired(t, 1, shape)
    ax, ay = desired(t, 2, shape)
    jx, jy = desired(t, 3, shape)
    ix = (J*ax+bx*vx)/(rm*Km)
    iy = (J*ay+by*vy)/(rm*Km)
    ux = Lm * (J * jx + bx * ax) / (rm * Km) + Rm * ix
    uy = Lm * (J * jy + by * ay) / (rm * Km) + Rm * iy

    return ux + uy, ux - uy, ix, iy


def run(shape):
    x0, y0 = desired(0, 0, shape)
    vx0, vy0 = desired(0, 1, shape)
    _, _, ix0, iy0 = input_voltages(0, shape)

    z0 = [x0, vx0, ix0, y0, vy0, iy0]

    def model(t, z):
        x, vx, ix, y, vy, iy = z

        ul, ur, _, _ = input_voltages(t, shape)
        ux = (ul+ur)/2
        uy = (ul-ur)/2

        return vx, (rm*Km*ix - bx*vx)/J, (ux - Rm*ix)/Lm,vy,(rm*Km*iy - by*vy)/J,(uy - Rm*iy)/Lm


    time = np.linspace(0, T, 801)
    sol = solve_ivp(model, (0, T), z0, t_eval=time)
    x_ref, y_ref = desired(time, 0, shape)
    plt.figure(figsize=(6, 5))
    plt.plot(sol.y[0]*1000, sol.y[3]*1000,
             label="Simulated trajectory")
    plt.plot(x_ref*1000, y_ref*1000, "--",
             label="Desired trajectory")

    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(shape.capitalize())
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.savefig(f"Q1_4_{shape}.png")


run("ellipse")
run("cardioid")