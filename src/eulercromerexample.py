import numpy as np

dt = 0.001
N = 10000
v = np.zeros(0, N)
a = np.zeros(0, N)
x = np.zeros(0, N)
t = np.linspace(0, N * dt)


def euler_cromer():
    for i in range(N):
        v[i + 1] = v[i] + a[i] * dt
        x[i + 1] = x[i] + v[i + 1] * dt
        t[i + 1] = t[i] + dt
    return v, x, t
