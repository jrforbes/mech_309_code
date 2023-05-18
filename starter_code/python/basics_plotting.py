"""MECH 309 python plotting basics.

J R Forbes, 2021/05/25
Based on
http://scipy-lectures.org/
https://numpy.org/
This script computes the step reponse of a mass-spring-damper system
# and plots the results
"""
# %%
# Packages
import numpy as np
# from scipy import linalg
from scipy import integrate
from matplotlib import pyplot as plt

# Main


def main():
    """Mass-spring-damper simulation."""
    # time
    dt = 1e-1
    t_start = 0
    t_end = 10
    t = np.arange(t_start, t_end, dt)

    # Initiate MassSpringDamper instance
    msd = MassSpringDamper(2, 10, 0.5)

    # Find step response
    y_step = msd.step_response(t)

    # Find impulse response
    y_impulse = msd.impulse_response(t)

    # Find response to arbitary intital condition (IC) by numerically
    # integrating the ODE
    y0 = np.array([0.1, -0.05])  # initial condition
    sol = integrate.solve_ivp(
        msd.ode,
        (t_start, t_end),
        y0,
        t_eval=t,
        rtol=1e-6,
        atol=1e-6,
        method='RK45',
    )

    # Plotting
    # Plotting parameters
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', size=14)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')

    # Plot step and impulse responses on same plot
    fig, ax = plt.subplots()
    # Format axes
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$y(t)$ (m)')
    # Plot data
    ax.plot(t, y_step, label='step response')
    ax.plot(t, y_impulse, label='impulse response')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

    # Plot step and impulse responses on different plots (one above the other)
    fig, ax = plt.subplots(2, 1)
    # Format axes
    for a in np.ravel(ax):
        a.set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$y_{\mathrm{step}}(t)$ (m)')
    ax[1].set_ylabel(r'$y_{\mathrm{impulse}}(t)$ (m)')
    # Plot data
    ax[0].plot(t, y_step, label='step response')
    ax[1].plot(t, y_impulse, label='impulse response', color='C1')
    fig.tight_layout()

    # Plot step and impulse responses on different plots (side-by-side)
    fig, ax = plt.subplots(1, 2, sharey=True)
    # Format axes
    for a in np.ravel(ax):
        a.set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$y_{\mathrm{step}}(t)$ (m)')
    ax[1].set_ylabel(r'$y_{\mathrm{impulse}}(t)$ (m)')
    # Plot data
    ax[0].plot(t, y_step, label='step response')
    ax[1].plot(t, y_impulse, label='impulse response', color='C1')
    fig.tight_layout()

    # Plot response to ICs on different plots
    fig, ax = plt.subplots(2, 1)
    # Format axes
    for a in np.ravel(ax):
        a.set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$y_{\mathrm{IC}}(t)$ (m)')
    ax[1].set_ylabel(r'$\dot{y}_{\mathrm{IC}}(t)$ (m/s)')
    # Plot data
    ax[0].plot(t, sol.y[0, :], label='position')
    ax[1].plot(t, sol.y[1, :], label='velocity', color='C2')
    ax[0].set_xlim([0, t[-1]/2])
    ax[0].set_ylim([-0.1, 0.5])
    fig.tight_layout()
    plt.show()
    # fig.savefig('figs/response.pdf')


# Classes


class MassSpringDamper:
    def __init__(self, mass, stiffness, damping):
        """Constructor for mass, spring, damper object.

        Parameters
        ----------
        mass : float
            Mass value, kg
        stiffness : float
            Stiffness value, N/m
        damping : float
            Damping value, (N*s)/ m
        """
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

    @property
    def _omega_n(self):
        return np.sqrt(self.stiffness / self.mass)  # natural frequency

    @property
    def _zeta(self):
        return self.damping / (2 * self.mass * self._omega_n)  # damping ratio

    @property
    def _omega_d(self):
        return self._omega_n * np.sqrt(1 - self._zeta**2)  # damped frequency

    def step_response(self, t):
        """Method for step response.

        Parameters
        ----------
        t : numpy.ndarray
            Time, seconds, shape = (N,)

        Returns
        -------
        numpy.ndarray :
            Time domain step response, m
        """
        y = (1 - 1 / np.sqrt(1 - self._zeta**2)
             * np.exp(-self._zeta * self._omega_n * t)
             * (np.sqrt(1 - self._zeta**2) * np.cos(self._omega_d * t)
                + self._zeta * np.sin(self._omega_d * t)))
        return y

    def impulse_response(self, t):
        """Method for impulse response.

        Parameters
        ----------
        t : numpy.ndarray
            Time, seconds, shape = (N,)

        Returns
        -------
        numpy.ndarray :
            Time domain impulse response, m
        """
        y = (self._omega_n / np.sqrt(1 - self._zeta**2)
             * np.exp(-self._zeta * self._omega_n * t)
             * np.sin(self._omega_n * np.sqrt(1 - self._zeta**2) * t))
        return y

    def ode(self, t, y):
        """Method for integration of mass, spring, damper ODE.

        y_dot = f(y) given y0

        Parameters
        ----------
        t : float
            Time, seconds,
        y : numpy.ndarray
            Input, units, (2, 1)

        Returns
        -------
        numpy.ndarray :
            y_dot, units, shape = (2, 1)
        """
        A = np.array([
            [0, 1],
            [-self.stiffness / self.mass, -self.damping / self.mass]
        ])
        y_dot = A @ y.reshape((-1, 1))
        return y_dot.ravel()  # flatten the array y_dot


if __name__ == '__main__':
    main()

# %%
