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

j = 0
while True:
    t = j*dt
    active = min(j + 1, N)
    if j <N:
        T[j] =T_nozzle
    times.append(t)
    history.append(T.copy())

    #Stop when the printed filament has a nearly uniform temperature
    if j >= N and T.max() - T.min() <= tol:
        break
    if j<N:
        #Open filament, no heat flux at either end
        segment = T[:active]
        left = np.r_[segment[0], segment[:-1]]
        right = np.r_[segment[1:], segment[-1]]
        T[:active] +=dt*(alpha*(left - 2*segment + right)/ds**2 - beta*(segment - T_eq))
    else:
        # Closed filament, periodic neighbours
        T += dt*(alpha*(np.roll(T, 1) - 2*T + np.roll(T, -1))/ds**2 - beta*(T - T_eq))
    j += 1

times = np.array(times)
history = np.array(history)
maximum = np.nanmax(history, axis=1)
minimum = np.nanmin(history, axis=1)
t_uniform = times[-1]

print(f"Loop length: {L:.4f} m; printing speed: {L/t_print:.4f} m/s")
print(f"Equilibrium temperature: {T_eq:.2f} deg C")
print(f"Near-uniform time: {t_uniform:.2f} s")
print(f"Time after deposition: {t_uniform-t_print:.2f} s")

fig, ax = plt.subplots(figsize=(8, 4.8))
early = times <= 65
im = ax.pcolormesh(s*1000, times[early], history[early], shading='nearest',
                   cmap='inferno', vmin=T_eq, vmax=T_nozzle)
ax.plot([0, L*1000], [0, t_print], 'c--', label='Deposition front')
ax.axhline(t_print, color='white', linestyle=':', label='Loop completed')
ax.set(xlabel='Arc length along ellipse (mm)', ylabel='Time (s)',
       title='Filament temperature during deposition and cooling')
ax.legend()
fig.colorbar(im, ax=ax, label='Filament temperature (°C)')
fig.tight_layout()
fig.savefig('Q1_6_temperature_map.png')

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(times, maximum, label='Maximum filament temperature')
ax.plot(times, minimum, label='Minimum filament temperature')
ax.axhline(T_eq, color='black', linestyle='--', label='Equilibrium temperature')
ax.axvline(t_print, color='gray', linestyle=':', label='Deposition completed')
ax.axvline(t_uniform, color='green', linestyle='--', label='Nearly uniform temperature')
ax.set(xlabel='Time since first deposition (s)', ylabel='Temperature (°C)',
       title='Cooling during and after deposition')
ax.legend()
fig.tight_layout()
fig.savefig('Q1_6_cooling_curves.png')


