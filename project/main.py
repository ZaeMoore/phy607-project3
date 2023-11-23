import numpy as np
import matplotlib.pyplot as plt
import tqdm
import emcee

def data(N):
    """Generate initial random spin data in a lattice structure"""
    return np.random.choice([-1, 1], size=(N, N))

def delta_energy(lattice, i, j, J):
    """Calculate the energy change for flipping a spin at (i, j)."""
    N = lattice.shape[0]
    spin = lattice[i, j]
    neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
    return 2 * J * spin * neighbors

def mcmc(lattice, beta, J, iterations):
    """Perform the MCMC simulation."""
    N = lattice.shape[0]
    magnetization = []
    energy = []

    for step in tqdm.tqdm(range(iterations)):
        i, j = np.random.randint(0, N, size=2)
        dE = delta_energy(lattice, i, j, J)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1

        magnetization.append(calculate_magnetization(lattice))
        energy.append(calculate_energy(lattice, J))
    return lattice, magnetization, energy

def calculate_magnetization(lattice):
    """Calculate the magnetization of the lattice."""
    return np.sum(lattice)

def calculate_energy(lattice, J):
    """Calculate the total energy of the lattice."""
    N = lattice.shape[0]
    energy = 0
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
            energy -= J * spin * neighbors
    return energy / 2  # Each pair counted twice

def specific_heat(energies, beta):
    """Calculate the specific heat."""
    energy_sq = np.square(energies)
    C = (np.mean(energy_sq) - np.mean(energies)**2) * beta**2
    return C

# Parameters
N = 100
num_steps = 100
J = 1
beta_values = np.linspace(0.2, 0.6, 10)

# Data containers
magnetizations = []
energies = []
specific_heats = []

for beta in tqdm.tqdm(beta_values):
    lattice, mag, energy = mcmc(data(N), beta, J, num_steps)
    magnetizations.append(np.mean(mag))
    energies.append(np.mean(energy))
    specific_heats.append(specific_heat(energy, beta))
    
# emcee sampler
sampler = emcee.EnsembleSampler(500, N*N, delta_energy)

# Visualization of the Final Spin Lattice
plt.figure()
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.title("Final Spin Lattice")
im = plt.imshow(lattice, cmap="magma")
plt.colorbar(im)
plt.show()
    
# Visualization of Magnetization
plt.figure(figsize=(6, 4))
plt.plot(beta_values, magnetizations)
plt.title("Average Magnetization")
plt.xlabel("Beta (1/T)")
plt.ylabel("Magnetization")
plt.show()

# Visualization of Energy
plt.figure(figsize=(6, 4))
plt.plot(beta_values, energies)
plt.title("Average Energy")
plt.xlabel("Beta (1/T)")
plt.ylabel("Energy")
plt.show()

# Visualization of Specific Heat
plt.figure(figsize=(6, 4))
plt.plot(beta_values, specific_heats)
plt.title("Specific Heat")
plt.xlabel("Beta (1/T)")
plt.ylabel("C")
plt.show()

