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
plt.legend(loc='upper left', fontsize='small')
plt.title("Energy over Time for each Temperature")
plt.xlabel("Time")
plt.ylabel("Energy (E)")
plt.show()

# Visualization of Magnetization over time for each Temp value
plt.figure()
for i in range(len(mag_plot)):
    plt.plot(mag_plot[i], label = "Temperature %.2f"%temperature_values[i])
plt.legend(loc='center left', fontsize='small')
plt.title("Magnetization over Time for each Temperature")
plt.xlabel("Time")
plt.ylabel("Magnetization (M)")
plt.show()

# Parameters for multiple chain analysis
num_chains = 5  # Define the number of independent chains
energies_all_chains = []
magnetizations_all_chains = []

# Running multiple independent chains and storing only energy and magnetization data
for chain in range(num_chains):
    lattice = data(N)  # Initialize each chain with a different lattice
    _, _, _, _, _, energies, magnetizations = mcmc(
        lattice, beta, J, num_steps, N, measurement_gap)
    energies_all_chains.append(energies)
    magnetizations_all_chains.append(magnetizations)

# Gelman-Rubin Diagnostic Function
def gelman_rubin(data):
    n = len(data[0])  # Number of samples per chain
    m = len(data)     # Number of chains
    chain_means = np.mean(data, axis=1)
    overall_mean = np.mean(chain_means)
    B_over_n = np.sum((chain_means - overall_mean)**2) / (m - 1)
    W = np.sum([np.var(chain, ddof=1) for chain in data]) / m
    var_plus = ((n - 1) / n) * W + B_over_n
    R_hat = np.sqrt(var_plus / W)
    return R_hat

# Apply Gelman-Rubin Diagnostic for Magnetization and Energy
R_hat_magnetization = gelman_rubin(magnetizations_all_chains)
R_hat_energy = gelman_rubin(energies_all_chains)

# Plotting the energy for each chain
plt.figure(figsize=(12, 6))
for i, energy_chain in enumerate(energies_all_chains):
    plt.plot(energy_chain, label=f'Chain {i+1}')
plt.xlabel('Steps')
plt.ylabel('Energy')
plt.title('Energy for Each Chain')
plt.legend()
plt.show()

# Plotting the magnetization for each chain
plt.figure(figsize=(12, 6))
for i, magnetization_chain in enumerate(magnetizations_all_chains):
    plt.plot(magnetization_chain, label=f'Chain {i+1}')
plt.xlabel('Steps')
plt.ylabel('Magnetization')
plt.title('Magnetization for Each Chain')
plt.legend()
plt.show()

print("Gelman-Rubin Diagnostic for Magnetization:", R_hat_magnetization)
print("Gelman-Rubin Diagnostic for Energy:", R_hat_energy)

