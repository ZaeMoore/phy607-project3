"""
Main script for project three, Markov Chain Monte Carlo simulation for phy 607

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from tqdm import tqdm
import emcee

def data(N):
    """Generate initial random spin data in a lattice structure

    Parameters
    ----------
    N : int
        Size of the spin lattice will be NxN

    Returns
    -------
    latticeSpins : 2D array of ints
        Lattice of spin states
    """
    return np.random.choice([-1, 1], size=(N, N))

def delta_energy(lattice, i, j, J, N):
    """Calculate the energy change for flipping a spin at (i, j). This is the likelihood function
    
    Parameters
    ----------
    lattice : 2D array of ints
        Lattice of spin states
    i : int
        Row of randomly chosen element in the lattice
    j : int
        Column of randomly chosen element in the lattice
    J : int
        Strength of interaction between neighbors
    
    Returns
    -------
    int
        Change in energy for flipping a randomly chosen spin state
    """
    spin = lattice[i, j]
    neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
    return 2 * J * spin * neighbors

def mcmc(lattice, beta, J, total_steps, N, measurement_gap):
    """Run the Markov Chain Monte Carlo simulation for a number of steps and record the results
    
    Parameters
    ----------
    lattice : 2D array of ints
        Lattice of spin states
    beta : float
        Value of beta = 1/KbT
    J : int
        Strength of interaction between neighbors
    total_steps : int
        Number of steps to be done in the mcmc
    N : int
        Size of the spin lattice is NxN
    measurement_gap : int
        Determines how often the mcmc measures physical properties of the system

    Returns
    -------
    lattice : 2D array of ints
        Updated lattice of spin states
    avg_magnetization : float
        Average magnetization of system
    avg_energy : float
        Average energy of system
    specific_heat_val : float
        Specific heat of system
    susceptibility : float
        Susceptibility of system

    """
    equilibration_steps = int(0.2 * total_steps)
    magnetizations = []
    energies = []

    for step in tqdm(range(total_steps)):
        i, j = np.random.randint(0, N, size=2)
        dE = delta_energy(lattice, i, j, J, N)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1
        if step >= equilibration_steps and step % measurement_gap == 0:
            energies.append(-J * np.sum(lattice * (np.roll(lattice, 1, axis=0) + np.roll(lattice, -1, axis=0) +
                                      np.roll(lattice, 1, axis=1) + np.roll(lattice, -1, axis=1))))
            magnetizations.append(np.sum(lattice))

    avg_energy = np.mean(energies)
    avg_magnetization = np.mean(magnetizations)
    specific_heat_val = (np.var(energies) / (N**2)) * (beta**2)
    susceptibility = (np.var(magnetizations) / (N**2)) * beta

    return lattice, avg_magnetization, avg_energy, specific_heat_val, susceptibility, energies, magnetizations

#Now do the emcee method here and compare
#Reference data_post.py from class
#Create the emcee sampler
#sampler = emcee.EnsembleSampler(500, order, delta_energy)

# Parameters
N = 32  # Increased lattice size for better resolution
num_steps = 1000000  # Increased total number of steps for better averaging
measurement_gap = 5  # Measurement gap introduced
J = 1
kB = 1  # Boltzmann constant
temperature_values = np.linspace(1.6, 2.8, 10)  # Adjusted temperature range
beta_values = 1 / (kB * temperature_values)

# Data containers
magnetizations = []
avg_energies = []
specific_heats = []
susceptibilities = []
energy_plot = []
mag_plot = []
lattice_list = []

for beta in tqdm(beta_values):
    lattice = data(N)
    lattice, avg_mag, avg_energy, spec_heat, susceptibility, energy_list, mag_list = mcmc(lattice, beta, J, num_steps, N, measurement_gap)
    magnetizations.append(avg_mag / (N**2))
    avg_energies.append(avg_energy / (N**2))
    specific_heats.append(spec_heat)
    susceptibilities.append(susceptibility)
    energy_plot.append(energy_list)
    mag_plot.append(mag_list)
    lattice_list.append(lattice)

# Visualization of the Final Spin Lattice
for i in range(len(beta_values)):
    plt.figure(figsize=[7.00, 3.50])
    plt.title("Final Spin Lattice for Temperature %s"%temperature_values[i])
    im = plt.imshow(lattice_list[i], cmap="magma")
    plt.colorbar(im)
    plt.show()

# Visualization of Energy vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, avg_energies, color='red')
plt.title("Average Energy vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Energy (E)")
plt.show()

# Visualization of Magnetization vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, magnetizations, color='red')
plt.title("Average Magnetisation vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetization (M)")
plt.show()

# Visualization of Specific Heat vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, specific_heats, color='red')
plt.title("Specific Heat vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("C")
plt.show()

# Visualization of Susceptibility vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, susceptibilities, color='red')
plt.title("Susceptibility vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Susceptibility (χ)")
plt.show()

# Visualization of Energy over time for each Temp value
plt.figure()
for i in range(len(energy_plot)):
    plt.plot(energy_plot[i], label = "Temperature %.2f"%temperature_values[i])
plt.legend(loc='upper right', fontsize='small')
plt.title("Energy over Time for each Temperature")
plt.xlabel("Time")
plt.ylabel("Energy (E)")
plt.show()

# Visualization of Magnetization over time for each Temp value
plt.figure()
for i in range(len(mag_plot)):
    plt.plot(mag_plot[i], label = "Temperature %.2f"%temperature_values[i])
plt.legend(fontsize='medium')
plt.title("Magnetization over Time for each Temperature")
plt.xlabel("Time")
plt.ylabel("Magnetization (M)")
plt.show()

