import numpy as np
import matplotlib.pyplot as plt


'''
To Do           - tau interactions
                - clean up constant variables into dicts or lists
'''


'''
Variables and Constants
'''
# # Constants
# Mass difference squared between two flavour states
delta_m_sq = {'eu': 7.53e-5, 'tu': 2.44e-3, 'te': 2.44e-3, 'ue': 7.53e-5, 'ut': 2.44e-3, 'et': 2.44e-3}     # eV**2
delta_m_e_mu_sq = 7.53e-5      # eV**2
delta_m_tau_mu_sq = 2.44e-3    # eV**2  # can be +ve or -ve
delta_m_tau_e_sq = 2.44e-3     # eV**2  # can be +ve or -ve
delta_m_mu_e_sq = delta_m_e_mu_sq      # eV**2
delta_m_mu_tau_sq = delta_m_tau_mu_sq    # can be +ve or -ve
delta_m_e_tau_sq = delta_m_tau_e_sq     # eV**2

# Angle between flavour states
sin_q_theta = {'eu': 0.846, 'tu': 0.92, 'te': 0.093, 'ue': 0.846, 'ut': 0.92, 'et': 0.093}
sin_sq_theta_e_tau = 0.093
sin_sq_theta_e_mu = 0.846
sin_sq_theta_mu_tau = 0.92      # actually > 0.92 but it varies on atmospheric values
sin_sq_theta_mu_e = sin_sq_theta_e_mu
sin_sq_theta_tau_e = sin_sq_theta_e_tau
sin_sq_theta_tau_mu = sin_sq_theta_mu_tau

G_f = 1.1663787e-5         # GeV-2
sin_sq_theta_w = 0.22290        # Weinberg angle
m_e = 0.511 * 1e-3     # GeV c-2
m_u = 105 * 1e-3       # GeV c-2
sigma_naught = 1.72e-45         # m**2 / GeV



'''
Oscillation Probabilities   -   p_oscill = (np.sin(2 * theta))**2 * (np.sin((delta_m_sq * L) / (4 * E)))**2
'''
class Oscillations:
    def __init__(self, distance=1e6, energy=10):
        self.prob_list = {
            'eu': [], 'et': [], 
            'ue': [], 'ut': [], 
            'te': [], 'tu': [], 
            'ee': [], 'uu': [],     # do these last as theyre calculated based on the previously calculated
            'tt': [],
        }

        self.prob_reduced = {
            'e': [],
            'u': [],
            't': []
        }

        self.prob_neut = []

        self.max_points = 1000      # Number of points
        self.E = energy
        self.L = distance    # Km
        self.x_range = np.linspace(0, self.L / self.E, self.max_points)
        theta_range = np.linspace(0, 2 * np.pi, self.max_points)
    

    def calculate(self):
        for x in self.x_range:
            # print(f"Calculated Probability of {self.max_points}")
            for change in self.prob_list.keys():
                if change not in ['ee', 'uu', 'tt']:
                    self.prob_list[change].append(Oscillations.prob(self, flavours=change, x=x))
            self.prob_list['ee'].append(1 - (self.prob_list['eu'][-1] * self.prob_list['et'][-1]))
            self.prob_list['uu'].append(1 - (self.prob_list['ue'][-1] * self.prob_list['ut'][-1]))
            self.prob_list['tt'].append(1 - (self.prob_list['te'][-1] * self.prob_list['tu'][-1]))

        self.prob_reduced['e'] = [self.prob_list['ee'], self.prob_list['eu'], self.prob_list['et']]
        self.prob_reduced['u'] = [self.prob_list['uu'], self.prob_list['ue'], self.prob_list['ut']]
        self.prob_reduced['t'] = [self.prob_list['tt'], self.prob_list['te'], self.prob_list['tu']]
        
        # plotting
        Oscillations.plot(self)
        plt.show()


    def plot(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Neutrino Oscillations')
        n = 0
        for initial in self.prob_reduced.values():
            if n == 0:
                for p in initial:
                    axs[n].plot(self.x_range, p)
                    axs[n].legend(['e to e', 'e to mu', 'e to tau'], loc=1)
            if n == 1:
                for p in initial:
                    axs[n].plot(self.x_range, p)
                    axs[n].legend(['mu to mu', 'mu to e', 'mu to tau'], loc=1)
            if n == 2:
                for p in initial:
                    axs[n].plot(self.x_range, p)
                    axs[n].legend(['tau to tau', 'tau to e', 'tau to mu'], loc=1)
            n += 1


    def prob(self, flavours, x):
        if flavours == 'eu':
            prob = sin_sq_theta_e_mu * np.square(np.sin(1.27 * delta_m_e_mu_sq * x / 4))
        elif flavours == 'et':
            prob = sin_sq_theta_e_tau * np.square(np.sin(1.27 * delta_m_e_tau_sq * x / 4))
        elif flavours == 'ue':
            prob = sin_sq_theta_e_mu * np.square(np.sin(1.27 * delta_m_mu_e_sq * x / 4))
        elif flavours == 'ut':
            prob = sin_sq_theta_mu_tau * np.square(np.sin(1.27 * delta_m_mu_tau_sq * x / 4))
        elif flavours == 'te':
            prob = sin_sq_theta_tau_e * np.square(np.sin(1.27 * delta_m_tau_e_sq * x / 4))
        elif flavours == 'tu':
            prob = sin_sq_theta_tau_mu * np.square(np.sin(1.27 * delta_m_tau_mu_sq * x / 4))
        return prob



'''
Cross Sections
'''
class CrossSections:
    def __init__(self, energy, lepton):
        E_v = energy
        sigma_naught = 1.72e-45

        s_e = sigma_naught * np.pi / G_f**2
        s_u = s_e * (m_u / m_e)

        sigma_naught_e = (2 * m_e * G_f**2 * E_v) / np.pi
        sigma_naught_u = (2 * m_u * G_f**2 * E_v) / np.pi

        if lepton == 'e':
            sigma_naught = sigma_naught_e
            self.cs_without_ms = {'e_e': [], 'E_E': [], 'E_U': [], 'u_e': [], 'u_u': [], 'U_U': []}
            self.cs_with_ms = {'e_e': [], 'E_E': [], 'E_U': [], 'u_e': [], 'u_u': [], 'U_U': []}
        
        elif lepton == 'u':
            sigma_naught = sigma_naught_u
            self.cs_without_ms = {'e_e': [], 'E_E': [], 'U_E': [], 'e_u': [], 'u_u': [], 'U_U': []}
            self.cs_with_ms = {'e_e': [], 'E_E': [], 'U_E': [], 'e_u': [], 'u_u': [], 'U_U': []}
        
        for flavour in self.cs_without_ms.keys():
            cs = CrossSections.neutrino_and_electron(self, flavour=flavour) * energy * sigma_naught
            self.cs_without_ms[flavour].append(cs)
            self.cs_with_ms[flavour].append(
                cs * energy * sigma_naught * 
                CrossSections.mass_suppression(
                    self, flavour, energy=energy, reaction_lepton=lepton
                )
            )

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
            cs = 1 / 12 - 1 / 3 * sin_sq_theta_w + 4 / 3 * np.square(sin_sq_theta_w)
        return cs

    def mass_suppression(self, flavour, energy, reaction_lepton='e'):
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
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        # neutrino-muon specific reactions
        elif flavour == 'U_E':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        elif flavour == 'e_u':
            zeta = 1 - ((m_e ** 2) / (m_in ** 2 + 2 * m_in * energy))
        return zeta



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
Gates
'''
class Gates:
    def __init__(self, energy_list):
        self.energy_list = energy_list
        self.gate_energy_reactions = None
        self.leptons = {'e': [], 'u': []}       # currently only considering e and mu (no tau)
        self.all_values = {}

    def calculate(self):
        for energy in self.energy_list:
            for lepton in self.leptons.keys():
                # print(f'\n Cross Sections for Reactions with {lepton} at {energy} GeV:')
                self.leptons[lepton] = CrossSections(energy=energy, lepton=lepton)    # Energy in GeV

            gate_reactions = Gates.gate_cs(Gates.combine_cs(self))
            if self.gate_energy_reactions is None:      # checking if first time running this script
                self.gate_energy_reactions = gate_reactions
            else:
                for flavour, value in gate_reactions.items():
                    self.gate_energy_reactions[flavour].append(value[0])
            self.all_values[energy] = [values for values in gate_reactions.values()]
            print(f"Average Prob for {energy} GeV: ", np.average(self.all_values[energy]))
            if energy == self.energy_list[-1]:
                print(f"Neutrino-Lepton Gate Probabilities for {energy}GeV: \n {gate_reactions}")
            
        # plot
        Gates.plot_energies(self)
        plt.show()


    def combine_cs(self):
        all_cs = {'e_e': [], 'e_u': [], 'u_u': [], 'u_e': [], 'E_E': [], 'E_U': [], 'U_U': [], 'U_E': []}
        e_class = self.leptons['e']
        u_class = self.leptons['u']
        for all_reaction in all_cs.keys():
            for e_reaction, e_value in e_class.cs_without_ms.items():
                if e_reaction == all_reaction:
                    all_cs[all_reaction].append(e_value[0])
            for u_reaction, u_value in u_class.cs_without_ms.items():
                if u_reaction == all_reaction:
                    all_cs[all_reaction].append(u_value[0])

        for all_reaction, all_values in all_cs.items():
            if len(all_values) > 1:
                all_cs[all_reaction] = [(all_values[0] + all_values[1]) / 2]
        return all_cs


    @staticmethod
    def gate_cs(all_cs):
        # creates our final dict with the probabilities for all gate interactions
        single_reactions = ['e_e', 'e_u', 'u_u', 'u_e']
        single_reactions_anti = ['E_E', 'E_U', 'U_U', 'U_E']
        whole_reaction = []
        for first_reaction in single_reactions:
            for second_reaction in single_reactions:
                whole_reaction.append(first_reaction[0] + second_reaction[0] + '_' + first_reaction[2] + second_reaction[2])
        for first_reaction in single_reactions_anti:
            for second_reaction in single_reactions_anti:
                whole_reaction.append(first_reaction[0] + second_reaction[0] + '_' + first_reaction[2] + second_reaction[2])
        gate_reactions = {whole_keys: [] for whole_keys in whole_reaction}

        
        # reaction 1 is reaction_combined[0] + '_' + reaction_combined[3]
        # reaction 2 in reaction_combined[1] + '_' + reaction_combined[4]
        for reaction_combined in gate_reactions.keys():
            first_reaction = reaction_combined[0] + '_' + reaction_combined[3]
            second_reaction = reaction_combined[1] + '_' + reaction_combined[4]
            gate_reactions[reaction_combined] = [all_cs[first_reaction][0] * all_cs[second_reaction][0]]

        # for key, value in gate_reactions.items():
        #     print(f"{key}: {value}")
        return gate_reactions


    def plot_energies(self):
        for flavour, values in self.gate_energy_reactions.items():
            self.gate_energy_reactions[flavour] = [values, list(self.energy_list)]

        for flavour, values in self.gate_energy_reactions.items():
            plt.plot(values[1], values[0], '-')
        plt.title(f"Neutrino-Lepton Gate Cross-Sections for Energies: {int(min(self.energy_list))}GeV - {int(max(self.energy_list))}GeV")
        plt.legend(self.gate_energy_reactions.keys())



'''     
Running
'''
Oscillations(distance=1e6, energy=10).calculate()    # Prints oscillation probabilities for flavours given the distance it travels

# WaveFunctions(accuracy=20)      # Currently Broken!!! Doesnt plot correctly # Accuracy is the number of points within the range

Gates(energy_list=np.linspace(1, 100, 10)).calculate()         # calculates and plots gate probabilities at various energies for different interactions

plt.show()



