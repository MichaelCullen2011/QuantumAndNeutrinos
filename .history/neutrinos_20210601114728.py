import numpy as np
import matplotlib.pyplot as plt


'''
To Do       -   Add graphing for a range of energies for each neutrino cs reaction
            -   Add neutrino-muon lepton reactions
'''


'''
Variables and Constants
'''
# Variables
delta_m_sq = 'Mass difference squared between two flavour states'
delta_m_e_mu_sq = 7.53e-5      # eV**2
delta_m_tau_mu_sq = 2.44e-3    # eV**2  # can be +ve or -ve
delta_m_tau_e_sq = 2.44e-3     # eV**2  # can be +ve or -ve
delta_m_mu_e_sq = delta_m_e_mu_sq      # eV**2
delta_m_mu_tau_sq = delta_m_tau_mu_sq    # can be +ve or -ve
delta_m_e_tau_sq = delta_m_tau_e_sq     # eV**2

sin_sq_theta_e_tau = 0.093
sin_sq_theta_e_mu = 0.846
sin_sq_theta_mu_tau = 0.92      # actually > 0.92 but it varies on atmospheric values
sin_sq_theta_mu_e = sin_sq_theta_e_mu
sin_sq_theta_tau_e = sin_sq_theta_e_tau
sin_sq_theta_tau_mu = sin_sq_theta_mu_tau


m_out = 10       # reactant mass out (electrons, muons etc)
m_in = 1000        # reactant mass in (electrons, muons etc)
E_v = 100     # Neutrino energy
L = 10       # distance oscillating (for oscillation)
E = E_v     # Neutrino beam energy
d = 10       # distance travelled (for decoherence)

G_f = 1.1663787e-5         # GeV-2
sin_sq_theta_w = 0.22290        # Weinberg angle
m_e = 0.511     # MeV c-2
m_u = 105       # MeV c-2
sigma_naught = 1.72e-45         # m**2 / GeV


mom_v = 0
# mom_e = m_e + v_e
# mom_u = m_u + v_u
mom_e = np.sqrt((sigma_naught * np.pi) / G_f**2)
mom_u = mom_e * (m_u / m_e)      # assume similar velocities

kinematic_e = (mom_v + mom_e)**2
kinematic_u = (mom_v + mom_u)**2

kinematic_e = m_e**2 + 2*m_e*E_v
kinematic_u = m_u**2 + 2*m_u*E_v

sigma_naught_e = (G_f**2 * kinematic_e) / np.pi
sigma_naught_u = (G_f**2 * kinematic_u) / np.pi


'''
Oscillation Probabilities   -   p_oscill = (np.sin(2 * theta))**2 * (np.sin((delta_m_sq * L) / (4 * E)))**2
'''


class Oscillations:
    def __init__(self, distance):
        p_e_mu_list = []
        p_e_tau_list = []
        p_e_e_list = []

        p_mu_e_list = []
        p_mu_tau_list = []
        p_mu_mu_list = []

        p_tau_e_list = []
        p_tau_mu_list = []
        p_tau_tau_list = []

        max = 1000      # Number of points
        E = 4      # GeV
        L = distance    # Km
        x_range = np.linspace(0, L / E, max)
        theta_range = np.linspace(0, 2 * np.pi, max)
        n = 0
        for x in x_range:
            n += 1
            print("Calculated Probability {} of {}".format(n, max))
            p_e_mu_list.append(Oscillations.prob(self, flavours='e_mu', x=x))
            p_e_tau_list.append(Oscillations.prob(self, flavours='e_tau', x=x))
            p_e_e_list.append(
                (1 - (Oscillations.prob(self, flavours='e_mu', x=x)) * (1 - Oscillations.prob(self, flavours='e_tau', x=x))))

            p_mu_e_list.append(Oscillations.prob(self, flavours='mu_e', x=x))
            p_mu_tau_list.append(Oscillations.prob(self, flavours='mu_tau', x=x))
            p_mu_mu_list.append(
                (1 - Oscillations.prob(self, flavours='mu_e', x=x)) * (1 - Oscillations.prob(self, flavours='mu_tau', x=x)))

            p_tau_e_list.append(Oscillations.prob(self, flavours='tau_e', x=x))
            p_tau_mu_list.append(Oscillations.prob(self, flavours='tau_mu', x=x))
            p_tau_tau_list.append(
                (1 - (Oscillations.prob(self, flavours='tau_e', x=x)) * (1 - Oscillations.prob(self, flavours='tau_mu', x=x))))

        prob_e = [p_e_e_list, p_e_mu_list, p_e_tau_list]
        prob_mu = [p_mu_mu_list, p_mu_e_list, p_mu_tau_list]
        prob_tau = [p_tau_tau_list, p_tau_e_list, p_tau_mu_list]

        prob_neut = [prob_e, prob_mu, prob_tau]

        fig, axs = plt.subplots(3)
        fig.suptitle('Neutrino Oscillations')
        n = 0
        for initial in prob_neut:
            if n == 0:
                for p in initial:
                    axs[n].plot(x_range, p)
                    axs[n].legend(['e to e', 'e to mu', 'e to tau'], loc=1)
            if n == 1:
                for p in initial:
                    axs[n].plot(x_range, p)
                    axs[n].legend(['mu to mu', 'mu to e', 'mu to tau'], loc=1)
            if n == 2:
                for p in initial:
                    axs[n].plot(x_range, p)
                    axs[n].legend(['tau to tau', 'tau to e', 'tau to mu'], loc=1)
            n += 1

    def prob(self, flavours, x):
        if flavours == 'e_mu':
            prob = sin_sq_theta_e_mu * np.square(np.sin(1.27 * delta_m_e_mu_sq * x / 4))
        elif flavours == 'e_tau':
            prob = sin_sq_theta_e_tau * np.square(np.sin(1.27 * delta_m_e_tau_sq * x / 4))
        elif flavours == 'mu_e':
            prob = sin_sq_theta_e_mu * np.square(np.sin(1.27 * delta_m_mu_e_sq * x / 4))
        elif flavours == 'mu_tau':
            prob = sin_sq_theta_mu_tau * np.square(np.sin(1.27 * delta_m_mu_tau_sq * x / 4))
        elif flavours == 'tau_e':
            prob = sin_sq_theta_tau_e * np.square(np.sin(1.27 * delta_m_tau_e_sq * x / 4))
        elif flavours == 'tau_mu':
            prob = sin_sq_theta_tau_mu * np.square(np.sin(1.27 * delta_m_tau_mu_sq * x / 4))
        return prob


'''
Cross Sections
'''


class CrossSections:
    def __init__(self, energy, lepton):
        if energy <= 11:
            print("Energy Value Lower than 11 GeV")
        else:
            E_v = energy
            kinematic_e = (m_e*1e3)**2 + 2*(m_e*1e3)*E_v    # mass in GeV c-2
            kinematic_u = (m_u*1e3)**2 + 2*(m_u*1e3)*E_v

            sigma_naught_e = (G_f**2 * kinematic_e) / np.pi
            sigma_naught_u = (G_f**2 * kinematic_u) / np.pi

            if lepton == 'e':
                sigma_naught = sigma_naught_e
                cs_without_ms = {'e_e': [], 'E_E': [], 'E_U': [], 'u_e': [], 'u_u': [], 'U_U': []}
                cs_with_ms = {'e_e': [], 'E_E': [], 'E_U': [], 'u_e': [], 'u_u': [], 'U_U': []}
            
            elif lepton == 'u':
                sigma_naught = sigma_naught_u
                cs_without_ms = {'e_e': [], 'E_E': [], 'U_E': [], 'e_u': [], 'u_u': [], 'U_U': []}
                cs_with_ms = {'e_e': [], 'E_E': [], 'U_E': [], 'e_u': [], 'u_u': [], 'U_U': []}
            
            for flavour in cs_without_ms.keys():
                cs = CrossSections.neutrino_and_electron(self, flavour=flavour) * energy * sigma_naught
                cs_without_ms[flavour].append(cs)
                cs_with_ms[flavour].append(
                    cs * energy * sigma_naught * 
                    CrossSections.mass_suppression(
                        self, flavour, energy=energy, reaction_lepton=lepton
                    )
                )
            print(cs_without_ms)
            print(cs_with_ms)

    def neutrino_and_electron(self, flavour):
        if flavour == 'e_e':
            cs = 0.25 + sin_sq_theta_w + (4 / 3) * np.square(sin_sq_theta_w)
        elif flavour == 'E_E':
            cs = (1 / 12) + 1 / 3 * (sin_sq_theta_w + 4 / 3 * np.square(sin_sq_theta_w))
        elif flavour == 'E_U' or flavour == 'U_E':
            cs = 1 / 3
        elif flavour == 'u_e' or flavour == 'e_u':
            cs = 1
        elif flavour == 'u_u':
            cs = 1 / 4 - sin_sq_theta_w + 4 / 3 * np.square(sin_sq_theta_w)
        elif flavour == 'U_U':
            cs = 1 / 12 - 1 / 3 * (sin_sq_theta_w + 4 / 3 * np.square(sin_sq_theta_w))
        return cs

    def mass_suppression(self, flavour, energy, reaction_lepton='e'):
        energy = energy * 1e3       # get energy in MeV
        m_e = 0.511     # MeV/c**2
        m_u = 105.7     # MeV/c**2
        m_E = m_e
        m_U = m_u
        if reaction_lepton == 'e':
            m_in = m_e
        elif reaction_lepton == 'u':
            m_in = m_u

        if flavour == 'e_e':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'E_E':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'E_U':
            zeta = 1 - ((m_u ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'u_e':
            zeta = 1 - ((m_u ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'u_u':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'U_U':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))#
        # neutrino-muon specific reactions
        elif flavour == 'U_E':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'e_u':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))

        print(zeta)
        return zeta

    def gate_cs(self, cs_with_ms):
        # single_reactions = cs_with_ms.keys()
        single_reactions = ['e_e', 'E_E', 'E_U', 'u_e', 'u_u', 'U_U']
        gate_reactions = []


'''
Wave Functions (for plotting)
'''


class WaveFunctions:
    def __init__(self, accuracy):
        phi = np.linspace(0, np.pi, accuracy)
        theta = np.linspace(0, 2 * np.pi, 20)
        prob_at_detector = {'e_e': [], 'e_mu': [], 'e_tau': [],
                            'mu_mu': [], 'mu_e': [], 'mu_tau': [],
                            'tau_tau': [], 'tau_e': [], 'tau_mu': []}
        flavour_list = ['e_e', 'e_mu', 'e_tau', 'mu_mu', 'mu_e', 'mu_tau', 'tau_tau', 'tau_e', 'tau_mu']
        detector_1 = {'e_e': [], 'e_mu': [], 'e_tau': [],
                      'mu_mu': [], 'mu_e': [], 'mu_tau': [],
                      'tau_tau': [], 'tau_e': [], 'tau_mu': []}
        detector_2 = {'e_e': [], 'e_mu': [], 'e_tau': [],
                      'mu_mu': [], 'mu_e': [], 'mu_tau': [],
                      'tau_tau': [], 'tau_e': [], 'tau_mu': []}

        for flavour_change in flavour_list:
            for phi_value in phi:
                prob_at_detector[flavour_change].append(WaveFunctions.prob(self, flavour=flavour_change, phi=phi_value))

            for tuple in prob_at_detector[flavour_change]:
                detector_1[flavour_change].append(tuple[0])
                detector_2[flavour_change].append(tuple[0])

        fig1 = plt.figure()
        fig2 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)
        ax1.title.set_text('Detector 1 Probabilities')
        ax2.title.set_text('Detector 2 Probabilities')

        ax1.set_xticks(ticks=np.linspace(start=0, stop=2 * np.pi, num=int(accuracy)))
        ax1.set_yticks(ticks=np.linspace(0, 1, num=5))
        ax2.set_xticks(ticks=np.linspace(start=0, stop=2 * np.pi, num=int(accuracy)))
        ax2.set_yticks(ticks=np.linspace(0, 1, num=5))


        for prob_list in detector_1.values():
            #print(prob_list)
            ax1.plot(phi, prob_list, '--', linewidth=1)

        for prob_list in detector_2.values():
            #print(prob_list)
            ax2.plot(phi, prob_list, '--', linewidth=1)

        ax1.legend(flavour_list, loc=1)
        ax2.legend(flavour_list, loc=1)

    def prob(self, flavour, phi):
        order_tau_d_same = 1
        order_tau_d_change = 1 / 3
        if flavour == 'e_e':
            prob_f1 = 1
            prob_f2 = 0
        elif flavour == 'e_mu':
            prob_f1 = 1 - (np.sqrt(sin_sq_theta_e_tau) / 2) * (1 - order_tau_d_change * np.cos(phi))
            prob_f2 = (np.sqrt(sin_sq_theta_e_tau) / 2) * (1 - order_tau_d_change * np.cos(phi))
        elif flavour == 'e_tau':
            prob_f1 = 1 - (np.sqrt(sin_sq_theta_e_tau) / 2) * (1 - order_tau_d_change * np.cos(phi))
            prob_f2 = (np.sqrt(sin_sq_theta_e_tau) / 2) * (1 - order_tau_d_change * np.cos(phi))
        if flavour == 'mu_mu':
            prob_f1 = 1
            prob_f2 = 0
        elif flavour == 'mu_e':
            prob_f1 = 1 - (np.sqrt(sin_sq_theta_e_mu) / 2) * (1 - order_tau_d_change * np.cos(phi))
            prob_f2 = (np.sqrt(sin_sq_theta_e_mu) / 2) * (1 - order_tau_d_change * np.cos(phi))
        elif flavour == 'mu_tau':
            prob_f1 = 1 - (np.sqrt(sin_sq_theta_mu_tau) / 2) * (1 - order_tau_d_change * np.cos(phi))
            prob_f2 = (np.sqrt(sin_sq_theta_mu_tau) / 2) * (1 - order_tau_d_change * np.cos(phi))
        if flavour == 'tau_tau':
            prob_f1 = 1
            prob_f2 = 0
        elif flavour == 'tau_e':
            prob_f1 = 1 - (np.sqrt(sin_sq_theta_tau_e) / 2) * (1 - order_tau_d_change * np.cos(phi))
            prob_f2 = (np.sqrt(sin_sq_theta_tau_e) / 2) * (1 - order_tau_d_change * np.cos(phi))
        elif flavour == 'tau_mu':
            prob_f1 = 1 - (np.sqrt(sin_sq_theta_tau_mu) / 2) * (1 - order_tau_d_change * np.cos(phi))
            prob_f2 = (np.sqrt(sin_sq_theta_tau_mu) / 2) * (1 - order_tau_d_change * np.cos(phi))
        return prob_f1, prob_f2


'''
Running
'''
# Oscillations(distance=1e6)    # Prints oscillation probabilities for flavours given the distance it travels

# WaveFunctions(accuracy=20)      # Currently Broken. Doesnt plot correctly # Accuracy is the number of points within the range

leptons = {'e': [], 'u': []}
for lepton in leptons.keys():
    # leptons[lepton] = CrossSections(energy=1000, lepton=lepton)    # Energy in GeV
    print(f'\n Cross Sections for Reactions with {lepton}:')
    CrossSections(energy=1000, lepton=lepton)    # Energy in GeV

plt.show()



