"""
Main script for project three, Markov Chain Monte Carlo simulation for phy 607

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import tqdm
import pymc3 as pm
import theano
import theano.tensor as tt

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

def delta_energy(lattice, i, j, J):
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
    N = lattice.shape[0]
    spin = lattice[i, j]
    neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
    return 2 * J * spin * neighbors

def mcmc(lattice, beta, J, iterations):
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
    """Calculate the magnetization of the lattice.
    
    Parameters
    ----------
    lattice : 2D array of ints
        Lattice of spin states

    Returns
    -------
    int
        Magnetization of the spin lattice
    """
    return np.sum(lattice)

def calculate_energy(lattice, J):
    """Calculate the total energy of the lattice.
    
        
    Parameters
    ----------
    lattice : 2D array of ints
        Lattice of spin states
    J : int
        Strength of interaction between neighbors

    Returns
    -------
    int
        Total energy of the spin lattice
    """
    N = lattice.shape[0]
    energy = 0
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            neighbors = lattice[(i-1)%N, j] + lattice[(i+1)%N, j] + lattice[i, (j-1)%N] + lattice[i, (j+1)%N]
            energy -= J * spin * neighbors
    return energy / 2  # Each pair counted twice

N = 32
num_steps = 100
lattice = data(N)
beta = 0.4
J = 1

finallattice, magnetization, energy = mcmc(lattice, beta, J, num_steps)


#pymc3
def get_H(spins):
    H = - (
        tt.roll(spins, 1, axis=1) * spins +
        tt.roll(spins, 1, axis=0) * spins +
        tt.roll(spins, -1, axis=1) * spins +
        tt.roll(spins, -1, axis=0) * spins
    )
    return H

def to_spins(lattice):
    return 2 * lattice - 1

def mc3_approach(beta, N, num_steps=10):
    shape = (N, N)
    x0 = np.random.randint(2, size=shape)
    with pm.Model() as model:
        x = pm.Bernoulli('x', 0.5, shape=shape, testval=x0)
        magnetization = pm.Potential('m', -get_H(to_spins(x)) * beta)
        scaling = .0006
        mul = int(N * N * 1.75)
        step = pm.BinaryMetropolis([x], scaling=scaling)
        trace = pm.sample(num_steps * mul * 5, step=step, chains=1, tune=False)
    lattice = 2 * trace[-1]['x'] - 1
    return lattice, trace

mc3_lattice, trace = mc3_approach(beta, N)


plt.figure()
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.title("Final Spin Lattice")
im = plt.imshow(finallattice, cmap="magma")
plt.colorbar(im)
plt.show()


plt.figure()
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.title("Final Spin Lattice pymc3")
im = plt.imshow(mc3_lattice, cmap="magma")
plt.colorbar(im)
plt.show()
plt.figure()
plt.plot(magnetization)
plt.title("Magnetization")

plt.figure()
plt.plot(energy)
plt.title("Energy")
