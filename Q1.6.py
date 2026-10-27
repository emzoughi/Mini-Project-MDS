import numpy as np
import matplotlib.pyplot as plt

#geometry of the printed ellipse and square filament cross-section
a, b = 0.04, 0.025
w = 0.0008

# filament properties and heat exchange with air and the heated bed
t_print = 8                   # time needed to deposit the entire ellipse
rho, cp, k = 1250, 1800, 0.2
h_air, h_bed = 10, 60
T_nozzle, T_air, T_bed = 465, 300, 333.15

#divide the loop into N cells, deposit one new cell per time step
N = 160
dt = t_print/N

# approximates the ellipse perimeter by summing distances between nearby points
#Computing it from a and b avoids introducing a separate arbitrary length
phi = np.linspace(0, 2*np.pi, 4001)
L = np.hypot(np.diff(a*np.cos(phi)), np.diff(b*np.sin(phi))).sum()
ds = L/N
s = (np.arange(N) + 0.5)*ds  #position of each cell centre along the filament

# coefficients of the heat equation dT/dt = alpha*d2T/ds2 - beta*(T-T_eq)
alpha = k/(rho*cp)
beta = (3*h_air + h_bed)/(rho*cp*w)
T_eq = (3*h_air*T_air + h_bed*T_bed)/(3*h_air + h_bed)

#nan marks material that has not yet been deposited
T = np.full(N, np.nan)
times, history = [], []
j = 0

while True:
    t = j*dt
    active = min(j + 1, N)     # number of cells already printed

    if j < N:
        T[j] = T_nozzle       # freshly deposited material enters at nozzle temperature

    # store the full temperature profile for plot
    times.append(t)
    history.append(T.copy())

    #stops only after the loop is complete and its temperature spread is <= 1 K
    if j >= N and T.max() - T.min() <= 1:
        break

    if j < N:
        #during deposition, heat diffuses only between already printed cells
        segment = T[:active]
        left = np.r_[segment[0], segment[:-1]]
        right = np.r_[segment[1:], segment[-1]]
        # repeated end values represent zero axial heat flux at the open ends
        T[:active] += dt*(alpha*(left - 2*segment + right)/ds**2
                          - beta*(segment - T_eq))
    else:
        #Once the ellipse closes, its first and last cells become neighbours
        # np.roll implements these periodic boundary conditions
        T += dt*(alpha*(np.roll(T, 1) - 2*T + np.roll(T, -1))/ds**2
                 - beta*(T - T_eq))

    j += 1

#extract the temperature range at every instant
times = np.array(times)
history = np.array(history)
maximum = np.nanmax(history, axis=1)  # Ignore cells not yet printed
minimum = np.nanmin(history, axis=1)
t_uniform = times[-1]

print(f'Loop length: {L:.4f} m; printing speed: {L/t_print:.4f} m/s')
print(f'Equilibrium temperature: {T_eq:.2f} K')
print(f'Near-uniform time: {t_uniform:.2f} s')
print(f'Time after deposition: {t_uniform-t_print:.2f} s')

#plot 1: temperature as a function of position along the loop and time.
plt.figure()
plt.pcolormesh(s*1000, times, history, shading='auto')
plt.colorbar(label='Filament temperature (K)')
plt.plot([0, L*1000], [0, t_print], '--', label='Deposition front')
plt.axhline(t_print, linestyle=':', label='Loop completed')
plt.xlabel('Arc length along ellipse (mm)')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('Q1_6_temperature_map.png')

#plot 2: cooling and convergence of the hottest and coldest printed cells
plt.figure()
plt.plot(times, maximum, label='Maximum temperature')
plt.plot(times, minimum, label='Minimum temperature')
plt.axhline(T_eq, linestyle='--', label='Equilibrium temperature')
plt.axvline(t_print, linestyle=':', label='Deposition completed')
plt.axvline(t_uniform, linestyle='--', label='Nearly uniform')
plt.xlabel('Time since first deposition (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.savefig('Q1_6_cooling_curves.png')
plt.show()
