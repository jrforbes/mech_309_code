"""MECH 309 python basics.

J R Forbes, 2021/05/19
Demoing ``def main"
"""
# %%
# Packages
import numpy as np
# from scipy import linalg
from matplotlib import pyplot as plt

#
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %%
# Functions


def pos(t, q0, v0, g):
    """Compute position given time, initial pos, initial vel, and g."""
    q = q0 + v0 * t - g * t ** 2
    return q


def vel(t, v0, g):
    """Compute velocity given time, initial vel, and g."""
    v = v0 - g * t
    return v


# %%
# Main
def main():
    """Particle falling in a gravity field."""
    q0 = 10  # m, initial position
    v0 = 10  # m/s, initial velocity
    g = 9.81  # m/s^2, gravity

    N = 50
    t = np.linspace(0, 10, N)
    x = np.zeros((2, N))  # store position and velocity
    for i in range(N):
        x[:, [i]] = np.array([[pos(t[i], q0, v0, g)],
                              [vel(t[i], v0, g)]])

    # Plotting
    # Plot position versus time
    fig, ax = plt.subplots()
    # Format axes
    ax.set_title(r'Position vs. Time')
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$q(t)$ (m)')
    # Plot data
    ax.plot(t, x[0, :], label='position', color='C0')
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Plot velocity versus time
    fig, ax = plt.subplots()
    # Format axes
    ax.set_title(r'Velocity vs. Time')
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$v(t)$ (m/s)')
    # Plot data
    ax.plot(t, x[1, :], label='velocity', color='C1')
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Plot position and velocity on one plot
    fig, ax = plt.subplots(2, 1)
    # Format axes
    for a in np.ravel(ax):
        a.set_xlabel(r'$t$ (s)')
    ax[0].set_title(r'Position vs. Time')
    ax[1].set_title(r'Velocity vs. Time')
    ax[0].set_ylabel(r'$q(t)$ (m)')
    ax[1].set_ylabel(r'$v(t)$ (m/s)')
    # Plot data
    ax[0].plot(t, x[0, :], label='position')
    ax[1].plot(t, x[1, :], label='velocity', color='C1')
    fig.tight_layout()
    # fig.savefig('figs/pos_vel_response.pdf')

    plt.show()


if __name__ == '__main__':
    main()

# %%
