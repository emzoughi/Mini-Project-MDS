import numpy as np
import matplotlib.pyplot as plt

a=0.04
b=0.025
w=0.0008
t_print=8
rho=1250
cp=1800
k=0.2
h_air = 10
h_bed = 60
T_nozzle = 210
T_air=25
T_bed= 60
N = 160
dt=t_print / N
tol= 1

phi = np.linspace(0, 2*np.pi, 4001)
L = np.hypot(np.diff(a*np.cos(phi)), np.diff(b*np.sin(phi))).sum()
ds = L/N
s = (np.arange(N) +0.5)*ds

alpha = k/(rho*cp)
beta = (3*h_air + h_bed)/(rho*cp*w)
T_eq = (3*h_air*T_air + h_bed*T_bed)/(3*h_air + h_bed)

T = np.full(N, np.nan)
times = []
history = []
