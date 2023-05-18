"""MECH 309 python basics.

J R Forbes, 2021/05/19
Based on
http://scipy-lectures.org/
https://numpy.org/
"""
# %%
# Packages
import numpy as np
from scipy import linalg
from scipy.stats import norm
import csv
from matplotlib import pyplot as plt

# %%
# Plotting
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %%
# Some slicing commands
z = np.linspace(0, 10, 10 + 1)
n_z = z.size
print('z is', z)

# First few z numbers
z_first_few = z[:3]
print('First few z numbers are', z_first_few)

# Last few z numbers
z_last_few = z[n_z - 3:]
print('Last few few z numbers are', z_last_few)

# Extract out every other element
z_even = z[::2]
print('Even z numbers are', z_even)

# Extract out every other element of starting at 1
z_odd = z[1::2]
print('Odd z numbers are', z_odd)

# Flip the data in the (N,) array
z_flip = z[::-1]
print('Flip z numbers are', z_flip, '\n\n')

# %%
# Matrices
# a list, vs a (N,) array, vs a (N, 1) array (a column), vs a (1, N) array
# (a row)
x0 = [1, 2]  # a list, don't use in scientific computing
x1 = np.array([1, 2])  # a (2,) array (also called a sequence)
x2 = np.array([[1], [2]])  # a (2, 1) array (i.e., a column matrix)
x3 = np.array([[1, 2]])  # a (1, 2) array (i.e., a row matrix)
print(x0, 'is of type', type(x0), 'with shape', np.shape(x0))
print(x1, 'is of type', type(x1), 'with shape', np.shape(x1))
print(x2, 'is of type', type(x2), 'with shape', np.shape(x2))
print(x3, 'is of type', type(x3), 'with shape', np.shape(x3), '\n\n')

# %%
# Extracting elements
# (3, 3) array, or a 3 x 3 matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
], dtype='float')

# a1 is an (3,) array, that being an array of numbers made up from
# the first column of A
a1 = A[:, 0]

# a2 is a (3, 1) array, a column matrix, that being the first column of A
a2 = A[:, [0]]

# This is (3, 2) array, a 3 x 2 matrix, composed of the first two columns of A
a3 = A[:, 0:2]

# convert (3,) array into a (3, 1) array (i.e., col matrix)
a4 = a1.reshape((3, 1))

# convert (3,) array into a (1, 3) array (i.e., row matrix)
a5 = a1.reshape((1, 3))

print(A)
print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print(a5[0, :-1])
print('Shape of A is', np.shape(A))
print('Shape of a1 is', np.shape(a1))
print('Shape of a4 is', np.shape(a4))

# %%
# swap rows
A[(0, 2), :] = A[(2, 0), :]
b = np.array([[1],
              [2],
              [1]])
A_tilde = np.hstack([A, b])  # see np.vstack, np.block, np.concatenate

# %%
# Transpose
A_trans = A.T  # transpose of a matrix
A_sym = 1 / 2 * (A + A_trans)  # matrix addition

# %%
# Complex matrices
Ac = np.array([
    [1 + 1j, -6 + 1j],
    [2 - 1j, 4]
], dtype=complex)

Ac_conj = np.conj(Ac)  # complex conjugate
Ac_herm = Ac_conj.T  # complex conjugate transpose
Ac_herm_other = np.conj(Ac).T

# %%
# Forming matrices
B = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print('Shape of B is', np.shape(B))

N = 3
c1 = np.linspace(0, N, N + 1)  # start, end, num-points
c2 = np.linspace(N, 0, N + 1)  # start, end, num-points
C = np.array([
    c1,
    c2,
])
print('Shape of c1 is', np.shape(c1), 'the shape of c2 is',
      np.shape(c2), 'and the shape of C is', np.shape(C))

d1 = np.arange(2, 10, 2)  # start, end (exclusive), step
print(type(d1), np.shape(d1))
# `-1` serves as a wildcard. If you want to reshape a (3,) array into (3, 1),
# you can do my_array.reshape((-1, 1)) and it will compute the
# unspecified dimension.
d1 = d1.reshape((-1, 1))
print(type(d1), np.shape(d1))

d2 = np.arange(8, 0, -2)  # start, end (exclusive), step
d2 = d2.reshape((-1, 1))
# note the (-1, 1) means wildcard, so, if d2 is (100,) then
# d2.reshape(-1, 1) will give you a (100, 1)

D = np.block([
    [d1, d2]
])
# or
# D = np.hstack((d1, d2))
# Use `np.block`, `np.hstack` and `np.vstack` to correctly form block matries

print('Shape of d1 is', np.shape(d1), 'the shape of d2 is',
      np.shape(d2), 'and the shape of D is', np.shape(D), '\n\n')

# %%
# Common matrices, such as identitiy, matrix full of ones,
# matrix full of zeros
id_matrix = np.eye(3)
zeros_matrix = np.zeros((3, 4))
ones_matrix = np.ones((3, 2))
diag_matrix = np.diag(np.arange(9))

# %%
# Matrix multiplication
# Generate a ramdom A matrix with a specific seed
np.random.seed(1234)
A = np.random.normal(0, 1, (3, 3))

x = np.array([
    [1],
    [2],
    [3],
])

b = A @ x

# %%
# Solving Ax = b
A = np.array([
    [9, 8, 7],
    [6, -5, 4],
    [-3, 2, -1],
])
b = np.array([
    [-1],
    [2],
    [-3],
])

# x = np.linalg.solve(A, b)
# print(x)
x = linalg.solve(A, b)  # preffered to np.linalg.solve(A, b)
print(x)

# %%
# Eigenvalues
eig_decomp = linalg.eig(A)
lam = eig_decomp[0]  # eigenvalues
V = eig_decomp[1]  # eigenvectors
#  print(lam)
#  print(V)
print('The eigenvalues are \n', lam, end='\n\n')
print('The eigenvectors are \n', V, end='\n\n')

# %%
# SVD
U_np, Sigma_np, V_np_trans = np.linalg.svd(A, full_matrices=True)

print('U from numpy is\n', U_np, '\n')
print('Sigma from numpy is\n', Sigma_np, '\n')
print('V from numpy is\n', V_np_trans.T, '\n\n')

# %%
# Random numbers

# Gaussian/normal distribution
np.random.seed(123321)
mu, sigma = 5.0, 2.0
normal_data = np.random.normal(mu, sigma, 100000)
mu_data, std_data = norm.fit(normal_data)
print('The mean and standard deviation of the data is',
      mu_data, 'and', std_data, '\n\n')

# print mean and std. dev.
print(np.mean(normal_data), np.std(normal_data), '\n\n')

# Plot data
N_samples = 100
fig, ax = plt.subplots()
count, bins, patches = plt.hist(normal_data, N_samples, density=True)
ax.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2
        / (2 * sigma**2)), linewidth=2, color='r')
ax.set_xlabel('x (units)')
ax.set_ylabel('Normalized Count')
fig.tight_layout()

# Uniform distribution
np.random.seed(1234321)
x_min, x_max = -2, 2
uniform_data = np.random.uniform(x_min, x_max, N_samples)
print(np.mean(uniform_data), np.std(uniform_data), '\n\n')

fig, ax = plt.subplots()
count, bins, patches = plt.hist(uniform_data, N_samples, density=True)
plt.plot(bins, np.ones_like(bins) / (x_max - x_min), linewidth=2, color='r')
ax.set_xlabel('x (units)')
ax.set_ylabel('Normalized Count')
fig.tight_layout()

plt.show()

# %%
# Save csv file
t = np.arange(0, 1, 0.1)
x = t ** 2
data_write = np.hstack((t.reshape(t.shape[0], -1), x.reshape(x.shape[0], -1)))
with open("data_file.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["Time", "x^2"])  # write header
    writer.writerows(data_write)

# alternative way
np.savetxt('data_file2.csv',
           data_write,
           fmt='%.8f',
           delimiter=',',
           header='t, x^2')

# %%
# Open csv file
data_read = np.loadtxt('data_file.csv',
                       dtype=float,
                       delimiter=',',
                       skiprows=1,
                       usecols=(0, 1,))

print(data_read)

# %%
