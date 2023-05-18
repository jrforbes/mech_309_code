"""MECH 309 python basics.

J R Forbes, 2021/05/19
Based on
http://scipy-lectures.org/
https://numpy.org/
"""
# %%
# Packages
import numpy as np
# from scipy import linalg
from matplotlib import pyplot as plt
import pathlib

# clear screen trick
# print("\033[H\033[J")
# print("\014")

# %%
# Basic addition, subtraction, division, etc.
x = 2  # variable definition
y = 3  # variable definition
print(x + y)  # x plus y
print(x - y)  # x subtract y
print(x / y)  # x divided by y
print(pow(x, y))  # x to the power of y
print(x**y)  # x to the power of y
print(x + y, end='\n\n')

# breakpoint() # type h, help, type c, continue, type q, quit debug

# %%
# Basic np commands
# https://numpy.org/doc/stable/reference/routines.math.html
print(np.pi)  # pi = 3.14159
print(np.sqrt(x))  # square root of x
print(np.exp(y))  # exp(y)
print(np.sin(np.pi / y))  # sin(pi / y)
print(np.arcsin(np.sin(np.pi / y)))
print(np.deg2rad(90))  # deg to rad
print(np.rad2deg(np.pi / 2))  # rad to deg
print(np.round([0.34, 0.57], decimals=1))  # round
print(np.floor(np.sqrt(x)))  # floor
print(np.ceil(np.sqrt(x)))  # ceil
print(np.trunc(np.sqrt(x)))  # trunc
xp = np.array([1, 2])
yp = np.array([5, 6])
print(np.interp(1.5, xp, yp))  # interpolate

# del x, y  # delete/erase x

# %%
# Basic logic and if-else statements
a = 4
b = 7

if a < b:
    print('a is less than b')
elif (a == b):
    print('a equals b')
else:
    print('a is greater than b')

if not (a < b):
    print('a is not less than b')

c = 5
if a == c:
    print('a equals c')
else:
    print('a does not equal c')

if a != c:
    print('a does not equal c')

# %%
# for loops
N = 5
a = 0
x = np.zeros(N,)  # preallocate space
print(x)
for i in range(N):
    a = a + i
    x[i] = 2**i

print(a, end='\n\n')
print(x, end='\n\n')
print(x.shape, end='\n\n')
# notice that x is a (N,) array, not a (1, N) array (i.e., not a 1 x N matrix)

a = 0
x = np.zeros((N, 1))  # preallocate space
print(x)
for i in range(N):
    a = a + i
    x[i, :] = 2**i

print(a, end='\n\n')
print(x, end='\n\n')
print(x.shape, end='\n\n')
# notice that x is not a (N,) array, it is a (N, 1) array
# (i.e., is a N x 1 matrix)

y = x.copy()  # Copy x
y = y[::-1, :]  # Flip the data in the (N, 1) array
print(y, end='\n\n')
print(y.shape, end='\n\n')

a = 0
x = np.zeros((1, N))  # preallocate space
print(x)
for i in range(N):
    a = a + i
    x[:, i] = 2**i

print(a, end='\n\n')
print(x, end='\n\n')
print(x.shape, end='\n\n')
# notice that x is not a (N,) array, it is a (1, N) array
# (i.e., is a 1 x N matrix)
x = x.ravel()  # ravel converts the 2D (1, N) array into a 1D (N,) array

# %%
# while loops
b = 1
tol = 1e-3
i = 0
while b > tol:
    b = b / 2
    i += 1

print('Number of iterations to reach b = ', b, ' is ', i)

# Same while loop but just storing each iterate in an array
b = [1]  # list, because we will use "append", which is only lists
tol = 1e-3
i = 0
while b[i] > tol:
    b.append(b[i] / 2)  # here we are using append
    i += 1  # increment counter
b = np.array(b)  # convert from list to array (numpy array)
print(b)
print('Number of iterations to reach b = ', b[-1], ' is ', i, '\n')

# %%
# List comprehension example
# This is a clever way to make a list with a for loop quickly
c = [(1 / i ** 2) for i in range(1, 5)]
print(c)
c_length = len(c)
print(c_length)

# %%
# Function example
# particle falling in a gravity field
q0 = 10  # m, initial position
v0 = 10  # m/s, initial velocity
g = 9.81  # m/s^2, gravity


def pos(t, q0, v0, g):
    """Compute position given time, ICs, and g."""
    q = q0 + v0 * t - g * t ** 2
    return q


def vel(t, v0, g):
    """Compute velocity given time, IC, and g."""
    v = v0 - g * t
    return v


N = 50
t = np.linspace(0, 10, N)
x = np.zeros((2, N))  # store position and velocity
for i in range(N):
    x[:, [i]] = np.array([[pos(t[i], q0, v0, g)],
                          [vel(t[i], v0, g)]])

# %%
# Plotting
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)

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
# fig.savefig(path.joinpath('pos_vel_response.pdf'))
# fig.savefig('pos_vel_response.pdf')

plt.show()
# %%
