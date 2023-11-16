import numpy as np

def initialize_lattice(N):
    """Initialize the lattice with spins randomly set to either +1 or -1."""
    return np.random.choice([-1, 1], size=(N, N))

def delta_energy(lattice, i, j, J, h):
    """Calculate the energy change for flipping a spin at (i, j)."""
    N = lattice.shape[0]
    spin = lattice[i, j]
    neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
    return 2 * J * spin * neighbors + 2 * h * spin

def metropolis_step(lattice, beta, J, h):
    """Perform one Metropolis-Hastings step."""
    N = lattice.shape[0]
    for _ in range(N**2):
        i, j = np.random.randint(0, N, size=2)
        dE = delta_energy(lattice, i, j, J, h)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1

def calculate_magnetization(lattice):
    """Calculate the magnetization of the lattice."""
    return np.sum(lattice)

def calculate_energy(lattice, J, h):
    """Calculate the total energy of the lattice."""
    N = lattice.shape[0]
    energy = 0
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
            energy -= J * spin * neighbors
    return energy / 2  # Each pair counted twice

# Parameters
N = 10  # Lattice size
J = 1   # Interaction strength
h = 0   # External magnetic field
beta = 0.4  # Inverse temperature

# Initialize lattice
lattice = initialize_lattice(N)

# Run the simulation
num_steps = 10000
for step in range(num_steps):
    metropolis_step(lattice, beta, J, h)

    if step % 1000 == 0:
        magnetization = calculate_magnetization(lattice)
        energy = calculate_energy(lattice, J, h)
        print(f"Step: {step}, Magnetization: {magnetization}, Energy: {energy}")
