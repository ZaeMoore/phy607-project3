"""
Markov Chain Monte Carlo simulation of 2D Ising Model

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from tqdm import tqdm
import pymc3 as pm
import theano.tensor as tt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def data(size, initialization):
    """Generate initial random spin data in a lattice structure

    Parameters
    ----------
    N : int
        Size of the spin lattice will be NxN
    initialization : int
        Determines the starting state of the lattice

    Returns
    -------
    latticeSpins : 2D array of ints
        Lattice of spin states
    """
    lattice_size = (size, size)
    if initialization == 1:
        return np.ones(lattice_size)
    elif initialization == -1:
        return -1 * np.ones(lattice_size)
    elif initialization == 0:
        return np.random.choice([-1, 1], size=lattice_size)
    else:
        print("Invalid option. Assuming random initial states")
        return np.random.choice([-1, 1], size=lattice_size)


def delta_energy(lattice, i, j, j_param, size):
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
    neighbors = (
        lattice[(i - 1) % size, j]
        + lattice[(i + 1) % size, j]
        + lattice[i, (j - 1) % size]
        + lattice[i, (j + 1) % size]
    )
    return 2 * j_param * spin * neighbors


def mcmc(lattice, beta, j_param, total_steps, size, measurement_gap):
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
def mcmc(lattice, beta, j_param, total_steps, size, measurement_gap):
    equilibration_steps = int(0.20 * total_steps)
    magnetizations = []
    energies = []
    center_spin_trace = []  # Initialize the list to store the trace

    for step in tqdm(range(total_steps)):
        i, j = np.random.randint(0, size, size=2)
        dE = delta_energy(lattice, i, j, j_param, size)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1
            if step % measurement_gap == 0:  # Record the spin at the center on measurement steps
                center_spin_trace.append(lattice[size//2, size//2])
        if step >= equilibration_steps and step % measurement_gap == 0:
            energies.append(
                -j_param
                * np.sum(
                    lattice
                    * (
                        np.roll(lattice, 1, axis=0)
                        + np.roll(lattice, -1, axis=0)
                        + np.roll(lattice, 1, axis=1)
                        + np.roll(lattice, -1, axis=1)
                    )
                )
            )
            magnetizations.append(np.sum(lattice))

    avg_energy = np.mean(energies)
    avg_magnetization = np.mean(magnetizations)
    specific_heat_val = (np.var(energies) / (size**2)) * (beta**2)
    susceptibility = (np.var(magnetizations) / (size**2)) * beta

    return (
        lattice,
        avg_magnetization,
        avg_energy,
        specific_heat_val,
        susceptibility,
        energies,
        magnetizations,
        center_spin_trace,  # Return the trace for autocorrelation analysis
    )
# pymc3
def get_Energy(spins):
    """Calculates the energy of the spin lattice at a particular state

    Parameters
    ----------
    spins : 2D array of ints
        Lattice of spin states

    Returns
    -------
    energy : int
        Energy of the state

    """
    energy = -(
        tt.roll(spins, 1, axis=1) * spins
        + tt.roll(spins, 1, axis=0) * spins
        + tt.roll(spins, -1, axis=1) * spins
        + tt.roll(spins, -1, axis=0) * spins
    )
    return energy


def to_spins(lattice):
    """Adapts the pymc3 spin lattice from 0s and 1s to -1s and +1s
    Pymc3 cannot efficiently make a lattice with negative values, so
    we have it create one of 1s and 0s and then manipulate it after

    Parameters
    ----------
    lattice : 2D array of ints
        Lattice of spin states in 0s and +1s

    Returns
    -------
    2D array of ints
        Lattice of spin states in +1s and -1s

    """
    return 2 * lattice - 1


def mc3_approach(beta, size, num_steps=10):
    """Pymc3 approach
    I wrote this while being repeatedly stabbed by tiny ink filled needles

    Parameters
    ----------
    beta : float
        Value of beta = 1/KbT
    num_steps : int
        Number of steps to be done in the mcmc
    N : int
        Size of the spin lattice is NxN

    Returns
    -------
    lattice : 2D array of ints
        Updated lattice of spin states
    trace : MultiTrace
        Spin lattice at every step

    """
    shape = (size, size)
    x0 = np.random.randint(2, size=shape)
    with pm.Model() as model:
        x = pm.Bernoulli("x", 0.5, shape=shape, testval=x0)
        magnetization = pm.Potential("m", -get_Energy(to_spins(x)) * beta)
        scaling = 0.0006
        mul = int(size * size * 1.75)
        step = pm.BinaryMetropolis([x], scaling=scaling)
        trace = pm.sample(num_steps * mul * 5, step=step, chains=1, tune=False)
    lattice = 2 * trace[-1]["x"] - 1
    return lattice, trace


# Parameters
size = 32  # Increased lattice size for better resolution
num_steps = 1000000  # Increased total number of steps for better averaging
measurement_gap = 5  # Measurement gap introduced
j_param = 1  # Interaction parameter
kb = 1  # Boltzmann constant
temperature_values = np.linspace(1.6, 2.8, 10)  # Adjusted temperature range
beta_values = 1 / (kb * temperature_values)

# Data containers
magnetizations = []
avg_energies = []
specific_heats = []
susceptibilities = []
energy_plot = []
mag_plot = []
lattice_list = []
mc3_lattice_list = []

initialization = int(
    input(
        "Initializing spin lattice state. Please choose initial state. \
            \n Type '1' for all spins to be set to +1. \n Type '-1' for all spins to be set to -1. \
            \n Type '0' for spins to be random. \n Initial state input: "
    )
)

# Testing different temperatures to find the phase transition
for beta in tqdm(beta_values):
    lattice = data(size, initialization)
    (
        lattice,
        avg_mag,
        avg_energy,
        spec_heat,
        susceptibility,
        energy_list,
        mag_list,
        center_spin_trace  # Add this variable to capture the new return value
    ) = mcmc(lattice, beta, j_param, num_steps, size, measurement_gap)
    magnetizations.append(avg_mag / (size**2))
    avg_energies.append(avg_energy / (size**2))
    specific_heats.append(spec_heat)
    susceptibilities.append(susceptibility)
    energy_plot.append(energy_list)
    mag_plot.append(mag_list)
    lattice_list.append(lattice)
    mc3_lattice, trace = mc3_approach(beta, size)
    mc3_lattice_list.append(mc3_lattice)


# Visualization of the Final Spin Lattice
for i in range(len(beta_values)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize= (7,4), constrained_layout=True)
    im = axes[0].imshow(lattice_list[i], cmap="magma", aspect="equal")
    axes[0].set_title("Handwritten MCMC")
    im = axes[1].imshow(mc3_lattice_list[i], cmap="magma", aspect="equal")
    axes[1].set_title("pymc3")

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    fig.suptitle("Final Lattice State at Temperature %.2f" % temperature_values[i])
    plt.show()


# Visualization of Energy vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, avg_energies, color="red")
plt.title("Average Energy vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Energy (E)")
plt.show()

# Visualization of Magnetization vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, magnetizations, color="red")
plt.title("Average Magnetisation vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetization (M)")
plt.show()

# Visualization of Specific Heat vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, specific_heats, color="red")
plt.title("Specific Heat vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("C")
plt.show()

# Visualization of Susceptibility vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, susceptibilities, color="red")
plt.title("Susceptibility vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Susceptibility (χ)")
plt.show()

# Visualization of Energy over time for each Temp value
plt.figure()
for i in range(len(energy_plot)):
    plt.plot(energy_plot[i], label="Temperature %.2f" % temperature_values[i])
plt.legend(loc="upper left", fontsize="small")
plt.title("Energy over Time for each Temperature")
plt.xlabel("Time")
plt.ylabel("Energy (E)")
plt.show()

# Visualization of Magnetization over time for each Temp value
plt.figure()
for i in range(len(mag_plot)):
    plt.plot(mag_plot[i], label="Temperature %.2f" % temperature_values[i])
plt.legend(loc="center left", fontsize="small")
plt.title("Magnetization over Time for each Temperature")
plt.xlabel("Time")
plt.ylabel("Magnetization (M)")
plt.show()

# Parameters for multiple chain analysis
num_chains = 5  # the number of independent chains
energies_all_chains = []
magnetizations_all_chains = []

# Running multiple independent chains and storing only energy and magnetization data
for chain in range(num_chains):
    lattice = data(size, initialization)
    _, _, _, _, _, energies, magnetizations, center_spin_trace = mcmc(
        lattice, beta, j_param, num_steps, size, measurement_gap
    )
    energies_all_chains.append(energies)
    magnetizations_all_chains.append(magnetizations)
    
def autocorrelation(trace, lag=1):
    n = len(trace)
    mean = np.mean(trace)
    var = np.var(trace)
    cov = np.mean((trace[:n-lag] - mean) * (trace[lag:] - mean))
    return cov / var
    
lags = range(1, 5000)  # Adjust the number of lags as necessary
autocorrs = [autocorrelation(center_spin_trace, lag=lag) for lag in lags]

plt.figure(figsize=(12, 6))
plt.plot(lags, autocorrs, marker='o')
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation of Center Spin at Different Lags")
plt.grid(True)
plt.show()


# Gelman-Rubin Diagnostic Function
def gelman_rubin(data):
    """Gelman-Rubin Diagnostic Function

    Parameters
    ----------
    data : Array of floats
        Array of either energy or magnetization data for every chain

    Returns
    -------
    r_hat : float

    """
    n = len(data[0])  # Number of samples per chain
    m = len(data)  # Number of chains
    # Mean of each chain and overall mean
    chain_means = np.mean(data, axis=1)
    overall_mean = np.mean(chain_means)
    # Between-chain variance and Within-chain variance
    b_over_n = np.sum((chain_means - overall_mean) ** 2) / (m - 1)
    W = np.sum([np.var(chain, ddof=1) for chain in data]) / m
    # Variance estimate
    var_plus = ((n - 1) / n) * W + b_over_n
    # Potential scale reduction factor
    r_hat = np.sqrt(var_plus / W)
    return r_hat
    
burnin = int(0.05 * num_steps) 
energies_no_burnin = [chain[burnin:] for chain in energies_all_chains]
magnetizations_no_burnin = [chain[burnin:] for chain in magnetizations_all_chains]

# Apply Gelman-Rubin Diagnostic for Magnetization and Energy
r_hat_magnetization = gelman_rubin(magnetizations_no_burnin)
r_hat_energy = gelman_rubin(energies_no_burnin)

# Plotting the energy for each chain
plt.figure(figsize=(12, 6))
for i, energy_chain in enumerate(energies_all_chains):
    plt.plot(energy_chain, label=f"Chain {i+1}")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("Energy for Each Chain")
plt.legend()
plt.show()

# Plotting the magnetization for each chain
plt.figure(figsize=(12, 6))
for i, magnetization_chain in enumerate(magnetizations_all_chains):
    plt.plot(magnetization_chain, label=f"Chain {i+1}")
plt.xlabel("Steps")
plt.ylabel("Magnetization")
plt.title("Magnetization for Each Chain")
plt.legend()
plt.show()

print("Gelman-Rubin Diagnostic for Magnetization:", r_hat_magnetization)
print("Gelman-Rubin Diagnostic for Energy:", r_hat_energy)
print("Simulation Complete")
