import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def data(N):
    return np.random.choice([-1, 1], size=(N, N))

def delta_energy(lattice, i, j, J, N):
    spin = lattice[i, j]
    neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
    return 2 * J * spin * neighbors

def mcmc(lattice, beta, J, total_steps, N, measurement_gap):
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

    return lattice, avg_magnetization, avg_energy, specific_heat_val, susceptibility

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

for beta in tqdm(beta_values):
    lattice = data(N)
    lattice, avg_mag, avg_energy, spec_heat, susceptibility = mcmc(lattice, beta, J, num_steps, N, measurement_gap)
    magnetizations.append(avg_mag / (N**2))
    avg_energies.append(avg_energy / (N**2))
    specific_heats.append(spec_heat)
    susceptibilities.append(susceptibility)

# Visualization of the Final Spin Lattice
plt.figure(figsize=[7.00, 3.50])
plt.title("Final Spin Lattice")
im = plt.imshow(lattice, cmap="magma")
plt.colorbar(im)
plt.show()

# Visualization of Energy vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, avg_energies, color='red')
plt.title("Energy vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Energy (E)")
plt.show()

# Visualization of Magnetization vs Temperature (scatter plot)
plt.figure()
plt.scatter(temperature_values, magnetizations, color='red')
plt.title("Magnetisation vs Temperature")
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

