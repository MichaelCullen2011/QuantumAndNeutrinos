'''
This py function can calculate and plot:    -   Wavefuntions of a particle in a box at various energy levels
                                            -   Calculate and plot the hydrogen orbital electron densities (to varying
                                                degrees of accuracy to speed up the calculation time)
'''

import matplotlib.pyplot as plt
import numpy as np

'''
Constants and Variables
'''
m_e = 9.11e-31      # kg
h = 6.626e-34       # K s-1

x_range = np.linspace(0, 1, 100)    # 100 points between 0 and 1
length = 1       # m


'''
Equations
'''


def psi(n, L, x):
    psi_value = np.sqrt(2 / L) * np.sin(n * np.pi * x/L)
    return psi_value


def psi_squared(n, L, x):
    psi_squared_value = np.square(psi(n, L, x))
    return psi_squared_value


'''
Calculations    -   Probabilities in a box
                    Probability of 1st orbital
'''


class Box:
    def __init__(self, max_energy):
        plt.figure(figsize=(8, 8))
        plt.suptitle("Wave Functions", fontsize=18)

        for n in range(1, max_energy + 1):       # Just do first 4 energy levels
            psi_squared_values = []
            psi_values = []
            for x in x_range:
                psi_squared_values.append(psi_squared(x=x, n=n, L=length))
                psi_values.append(psi(x=x, n=n, L=length))

            both_psi = [psi_values, psi_squared_values]

            # Plot psi
            for y in both_psi:
                plt.subplot(max_energy, 2, 2 * n - 1)
                plt.plot(x_range, y)
                plt.legend(["Psi", "Psi**2"])
                plt.xlabel("L", fontsize=13)
                plt.ylabel("Psi and Psi*Psi", fontsize=13)
                plt.xticks(np.arange(0, 1, step=0.5))
                plt.title("Energy level: n = " + str(n), fontsize=16)
                plt.grid()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


class Orbitals:
    def __init__(self, orbital, accuracy):
        shell = orbital
        distance_values = []
        probability_values = []
        n = 0
        x, y, z, theta, psi, num = self.get_ranges(orbital=shell, num=accuracy)
        for x_value in x:
            n += 1
            print("Calculating {n_value} / {max}".format(n_value=n, max=num))
            for y_value in y:
                for z_value in z:
                    # Get orbital prob values
                    if orbital == '1s':
                        distance_values.append(str((x_value,
                                                    y_value,
                                                    z_value)))
                        probability_values.append(self.prob_1s(x=x_value,
                                                               y=y_value,
                                                               z=z_value))
                    elif orbital == '2s':
                        distance_values.append(str((x_value,
                                                    y_value,
                                                    z_value)))
                        probability_values.append(self.prob_2s(x=x_value,
                                                               y=y_value,
                                                               z=z_value))
                    elif orbital == '2p':
                        for theta_value in theta:
                            distance_values.append(str((x_value,
                                                        y_value,
                                                        z_value,
                                                        theta_value)))
                            probability_values.append(self.prob_2p(x=x_value,
                                                                   y=y_value,
                                                                   z=z_value,
                                                                   theta=theta_value))
                    elif orbital == '2px':
                        for theta_value in theta:
                            for psi_value in psi:
                                distance_values.append(str((x_value,
                                                            y_value,
                                                            z_value,
                                                            theta_value,
                                                            psi_value)))
                                probability_values.append(
                                    self.prob_2p_xy(x=x_value,
                                                    y=y_value,
                                                    z=z_value,
                                                    theta=theta_value,
                                                    psi=psi_value,
                                                    x_y='x'))
                    elif orbital == '2py':
                        for theta_value in theta:
                            for psi_value in psi:
                                distance_values.append(str((x_value,
                                                            y_value,
                                                            z_value,
                                                            theta_value,
                                                            psi_value)))
                                probability_values.append(self.prob_2p_xy(x=x_value,
                                                                          y=y_value,
                                                                          z=z_value,
                                                                          theta=theta_value,
                                                                          psi=psi_value,
                                                                          x_y='y'))
                    elif orbital == '3s':
                        distance_values.append(str((x_value,
                                                    y_value,
                                                    z_value)))
                        probability_values.append(self.prob_3s(x=x_value,
                                                               y=y_value,
                                                               z=z_value))
                    elif orbital == '3p':
                        for theta_value in theta:
                            distance_values.append(str((x_value,
                                                        y_value,
                                                        z_value,
                                                        theta_value)))
                            probability_values.append(self.prob_3p(x=x_value,
                                                                   y=y_value,
                                                                   z=z_value,
                                                                   theta=theta_value))
        print(probability_values)
        probability_values = probability_values / sum(probability_values)

        coordinates = np.random.choice(distance_values, size=100000, replace=True, p=probability_values)
        value_mat = [i.split(',') for i in coordinates]
        value_mat = np.array(value_mat)

        x_coords = [float(i.item()[1:]) for i in value_mat[:, 0]]
        y_coords = [float(i.item()) for i in value_mat[:, 1]]
        z_coords = [float(i.item()[0: -1]) for i in value_mat[:, 2]]

        # Plotting
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, alpha=0.05, s=2)
        ax.set_title("Hydrogen {orbital} Orbital Density. Accuracy = {accuracy}".format(orbital=orbital, accuracy=num))

    def get_ranges(self, orbital, num):
        if orbital == '1s':
            num = 50
        if orbital == '2s':
            num = 50
        elif orbital == '2p':
            num = 20
        elif orbital == '2px':
            num = 10
        elif orbital == '2py':
            num = 10
        elif orbital == '3s':
            num = 30
        elif orbital == '3p':
            num = 20
        x = np.linspace(0, 1, num)
        y = np.linspace(0, 1, num)
        z = np.linspace(0, 1, num)
        theta = np.linspace(0, 2 * np.pi, num)
        psi = np.linspace(0, np.pi, num)
        return x, y, z, theta, psi, num

    def prob_1s(self, x, y, z):
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        prob = np.square(np.exp(-r) / np.sqrt(np.pi))
        return prob

    def prob_2s(self, x, y, z):
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        prob = np.square((2 - r) * (np.exp(-r / 2)) / np.sqrt(32 * np.pi))
        return prob

    def prob_2p(self, x, y, z, theta):
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        prob = np.square(((r * np.exp(-r / 2)) * np.cos(theta)) / (np.sqrt(32 * np.pi)))
        return prob

    def prob_2p_xy(self, x, y, z, theta, psi, x_y='x'):
        a = 1
        if x_y == 'x':
            a = 1
        elif x_y == 'y':
            a = -1
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        prob = np.square(((r * np.exp(-r / 2)) * np.sin(theta)) * np.exp(a * np.imag(1) * psi) / (np.sqrt(64 * np.pi)))
        return prob

    def prob_3s(self, x, y, z):
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        prob = np.square(((np.exp(-r / 3)) * (27 - 18 * r + 2 * np.square(r))) / (81 * np.sqrt(3 * np.pi)))
        return prob

    def prob_3p(self, x, y, z, theta):
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        prob = np.square((6 * r - np.square(r)) * (np.exp(-r / 3) * np.cos(theta) * np.sqrt(2 / np.pi)) / 81)
        return prob


'''
Run Section
'''
Box(max_energy=3)
# Orbitals(orbital='1s', accuracy=50)      # Can also include an accuracy value. default = 50
# Orbitals(orbital='2s', accuracy=50)      # for fast calc: accuracy = 50
# Orbitals(orbital='2p', accuracy=20)      # for fast calc: accuracy = 20
# Orbitals(orbital='2px', accuracy=10)      # for fast calc: accuracy = 10
# Orbitals(orbital='2py', accuracy=10)      # for fast calc: accuracy = 10
# Orbitals(orbital='3s', accuracy=30)      # for fast calc: accuracy = 30
# Orbitals(orbital='3p', accuracy=20)      # for fast calc: accuracy = 20

plt.show()









