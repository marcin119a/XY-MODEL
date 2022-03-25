import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.stats import gamma, uniform
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

"""
 applying Metropolis algorithm
 input: T/temperature
        S/spins configuration(in 1d list)
        H/external field.default value=0
"""
types = 8
type_fixed = 0


class XYSystem():
    def __init__(self, temperature= 3, width=10):
        self.width = width
        self.num_spins = width**2
        self.support_end = 5
        L, N = self.width, self.num_spins
        self.nbr = {i: ((i // L) * L + (i + 1) % L, (i + L) % N,
                    (i // L) * L + (i - 1) % L, (i - L) % N) \
                                            for i in list(range(N))}
        self.thetas = gamma.rvs(np.ones((self.num_spins, 8)), loc=9, scale=0.5)
        self.spin_config = self._transf(self.thetas[:, 0], -pi, pi, 0, self.support_end)
        self.temperature = temperature
        self.energy = np.sum(self.get_energy()) / self.num_spins
        self.M = []
        self.Cv = []

    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def sweep(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            energy_i = -sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx])
            d_theta = np.random.uniform(-pi, pi)
            spin_temp = self.spin_config[idx] + d_theta
            energy_f = -sum(np.cos(spin_temp-self.spin_config[n]) for n in self.nbr[idx]) 
            delta_E = energy_f - energy_i
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += d_theta

    """ 
    calculate the energy of a given configuration  
    input: S/spin configuration in list
             H/external field, defult 0
    """
    def get_energy(self):
        energy_=np.zeros(np.shape(self.spin_config))
        idx = 0
        for spin in self.spin_config: #calculate energy per spin
            energy_[idx] = -sum(np.cos(spin-self.spin_config[n]) for n in self.nbr[idx])#nearst neighbor of kth spin
            idx += 1
        return energy_
        
    """
    Let the system evolve to equilibrium state
    """
    def equilibrate(self, max_nsweeps=int(1e4), temperature=None, H=None, show = False):
        if temperature is not None:
            self.temperature = temperature
        dic_thermal_t = {'energy': []}
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep()
            energy = np.sum(self.get_energy())/self.num_spins/2
            dic_thermal_t['energy'] += [energy]
            if show & (k%1e3 ==0):
                print('#sweeps=%i'% (k+1))
                print('energy=%.2f'%energy)
                self.show()
                self.show_map(text='Start equilibrate')
            if ((abs(energy-energy_temp)/abs(energy)<1e-4) & (k>500)) or k == max_nsweeps-1:
                print('\nequilibrium state is reached at T=%.1f'%self.temperature)
                print('#sweep=%i'%k)
                print('energy=%.2f'%energy)
                if show:
                    self.show()
                    self.show_map(text='End equilibrate')
                break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy = np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy = energy
        energy2 = np.average(np.power(dic_thermal_t['energy'][int(nstates/2):],2))
        self.Cv = (energy2-energy**2)*beta**2

    """
    Removing multiple angles for spins
    """
    def get_spins(self, degree=False):
        x = np.cos(self.spin_config)
        y = np.sin(self.spin_config)
        spins_norm = np.arctan2(y, x)
        return spins_norm

    def _transf(self, t, c, d, a, b):
        return c + ((d-c)/(b-a)) * (t - a)

    def inv_hgs(self):
        return self._transf(self.get_spins(), 0, 5, -pi, pi)


    """
    To see thermoquantities evolve as we cooling the systems down
    input: T_inital: initial tempreature
           T_final: final temperature
           sample/'log' or 'lin',mean linear sampled T or log sampled( centered at critical point)
    """
    def annealing(self, t_init=2.5, t_final=0.1, nsteps= 20, show_equi=False):
        # initialize spins. Orientations are taken from 0 - 2pi randomly.
        # initialize spin configuration
        dic_thermal = {
            'temperature': list(np.linspace(t_init, t_final, nsteps)),
            'energy': [],
            'Cv': []
        }
        for T in dic_thermal['temperature']:
            self.equilibrate(temperature=T)
            if show_equi:
                self.show_map()
            dic_thermal['energy'] += [self.energy]
            dic_thermal['Cv'] += [self.Cv]
        plt.plot(dic_thermal['temperature'], dic_thermal['Cv'],'.')
        plt.ylabel(r'$C_v$')
        plt.xlabel('T')
        plt.show()
        plt.plot(dic_thermal['temperature'], dic_thermal['energy'],'.')
        plt.ylabel(r'$\langle E \rangle$')
        plt.xlabel('T')
        plt.show()
        return dic_thermal

    """
    convert configuration inz list to matrix form
    """
    @staticmethod
    def list2matrix(s):
        n = int(np.size(s))
        l = int(np.sqrt(n))
        s = np.reshape(s, (l, l))
        return s

    """
    visualize a configuration
    input：S/ spin configuration in list form
    """
    def show(self, colored=False, text = None):
        config_matrix = self.list2matrix(self.spin_config)
        x, y = np.meshgrid(np.arange(0, self.width ), np.arange(0, self.width))
        u = np.cos(config_matrix)
        v = np.sin(config_matrix)
        plt.figure(figsize=(4, 4), dpi=100)
        Q = plt.quiver(x, y, u, v, units='width')
        plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',  coordinates='figure')
        plt.title('T=%.2f'%self.temperature+', #spins='+str(self.width)+'x'+str(self.width) + f'{text}')
        plt.axis('off')
        plt.savefig(f'{random.randint(1,100)}.png')

    def show_map(self, text=""):
        fig = plt.figure(figsize=(20, 10))
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.set_title(f"T={self.temperature}, {text}", fontsize=25)
        im0 = ax0.imshow(self.list2matrix(self.inv_hgs()),  vmin=0, vmax=self.support_end)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="10%", pad=0.05)

        fig.colorbar(im0, cax=cax0)
        plt.savefig(f'{random.randint(1,100)}.png')
