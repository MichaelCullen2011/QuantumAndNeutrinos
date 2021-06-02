import numpy as np
import sympy as sym
from sympy.plotting import plot, plot3d


'''
Constants and Variables
'''
m_e = {'kg': 9.11e-31, 'eV c2': 0.511e6}      # kg
h = {'m2 kg s-1': 6.62607004e-34, 'eV s': 4.1357e-15}
h_bar = {'m2 kg s-1': 1.0545718e-33, 'eV s': 6.5821e-16}

m_particle = m_e['eV c2']
L = H = W = 1       # in m
x = sym.Symbol('x')
y = sym.Symbol('y')
z = sym.Symbol('z')
r = sym.Symbol('r')
theta = np.linspace(0, 2 * np.pi, 20)
phi = np.linspace(0, np.pi, 20)


'''
Functions
'''


def laplace_sq(func):
    # diff_x_sq = sym.diff(sym.diff(func, x))
    # diff_y_sq = sym.diff(sym.diff(func, y))
    # diff_z_sq = sym.diff(sym.diff(func, z))
    delta_sq = sym.diff(sym.diff(func, x)) + sym.diff(sym.diff(func, y)) + sym.diff(sym.diff(func, z))
    return delta_sq


'''
Equations
'''
psi_a = sym.sin(x)     # example wavefunctions
psi_b = sym.cos(y)
psi_c = sym.sin(x)**2 + sym.cos(y)**2

V_0 = 0             # for potential-wells


# Hamiltonian in one dimension
def hamiltonian_one_dim(func, potential, var=x, constants=False):
    if constants:
        ham = - (h_bar['eV s']**2 / 2 * m_particle) * sym.diff(sym.diff(func, var), var) + potential
    else:
        ham = - sym.diff(sym.diff(func, var), var)
    return ham


# Hamiltonian in three dimensions
def hamiltonian_three_dim(func, potential, constants=False):
    if constants:
        ham = - (h_bar['eV s']**2 / 2 * m_particle) * laplace_sq(func) + potential
    else:
        ham = - laplace_sq(func)
    return ham


# print(hamiltonian_one_dim(func=psi_a, potential=V_0, var=x, constants=False))
# print(hamiltonian_three_dim(func=psi_c, potential=V_0, constants=False))


'''
SHO
'''
# Variables
m = m_e['eV c2']
energy = {'J': 1.60218e-19, 'eV': 5}      # in eV
omega = energy['eV'] / h_bar['eV s']
r_sq = x**2 + y**2

# Equations
sho_potential_1D = m * omega**2 * x**2 / 2
sho_potential_3D = m * omega**2 * r_sq / 2

sho_ham_1D = hamiltonian_one_dim(func=psi_a, potential=sho_potential_1D, var=x, constants=True)
sho_ham_3D = hamiltonian_three_dim(func=psi_c, potential=sho_potential_3D, constants=True)

print(sho_ham_1D)
print(sho_ham_3D)

# Plotting
sho_ham_1D_plot = plot(sho_ham_1D, plot=False)
sho_ham_3D_plot = plot3d(sho_ham_3D, plot=False)


'''
Euler–Lagrange
'''


def lagrangian(kinetic, potential):
    l = kinetic - potential
    return l


def action(lagra, var, var_1, var_2):
    s = sym.integrate(lagra, (var, var_1, var_2))
    return s



