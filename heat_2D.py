import numpy as np
from numba import njit, prange, jit
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# Boundary condition
@njit(parallel=True, fastmath=True)
def boundary_condition(u, u_initial):

    n_x, n_y = u.shape
    center_x, center_y = n_x // 2, n_y // 2

    for x in prange(1, n_x - 1):
        for y in prange(1, n_y - 1):

            # if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
            if ((x - center_x)**2 + (y - center_y)**2 - radius)**3  <= (x - center_x)**2 * (y - center_y)**3:
                u[x, y] = u_initial
    return u

# Time evolution of the heat
@njit(parallel=True, fastmath=True, nogil=True)
def heat_evolve(u, dt, dx, dy):

    un = u.copy()
    n_x, n_y = u.shape

    for x in prange(1, n_x - 1):
        for y in prange(1, n_y - 1):

            u[x, y] = un[x, y] + dt * ((un[x - 1, y] + un[x + 1, y] + un[x, y - 1] + un[x, y + 1] - 4. * un[x, y]) / (dx * dy))

    return u

@jit
def animate(u, u_initial):

    ims = []
    fig = plt.figure()

    for t in range(n_time):
        u = boundary_condition(u=u, u_initial=u_initial)

        im = plt.imshow(u.T, animated=True, origin='lower')
        ims.append([im])
        u = heat_evolve(u=u, dt=dt, dx=dx, dy=dy)

    ani = animation.ArtistAnimation(fig, ims, interval=5)
    ani.save('heat_evolution.mp4')




# Finite difference
dt = 0.005
dx, dy = 0.4, 0.4

# Number of discretization points
n_x, n_y = 250, 250
n_time = 2000

# Initialize values
u = np.zeros((n_x, n_y))
u_initial = 10.
radius = 5000.


animate(u, u_initial)


