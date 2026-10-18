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

