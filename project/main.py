"""
Main script for project three, Markov Chain Monte Carlo simulation for phy 607

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import tqdm

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
    latticeSpins = np.zeros(shape=(N, N))

    for i in latticeSpins:
        for j in i:
            latticeSpins[i,j] = np.random.choice([-1, 1])

    return latticeSpins

def new_array(s, N):
    """Generate potential new spin lattice to test

    Parameters
    ----------
    s : 2D array of ints
        Current data to be modified
    N : int
        Size of the spin lattice will be NxN

    Returns
    -------
    s : 2D array of ints
        New spin lattice to test
    """
    #s is the spins array
    randomRow = np.random.choice(range(0, N-1))
    randomColumn = np.random.choice(range(0, N-1))
    s[randomRow, randomColumn] *= -1
    return s
    #Given the old array, pick a random spin entry to flip and return the new array

def energy(s, J, N):
    """Calculate the energy of a state of spins

    Parameters
    ----------
    s : 2D array of ints
        Lattice of spin states

    Returns
    -------
    energy : int
        Energy of the state
    """
    
    atomEnergy = [] #Energy for each atom is that atom's spin state times the spin state of its neighbors
    for i in s:
        for j in i:
            atomEnergy.append(s[i,j] * (s[i,j+1] + s[i,j-1] + s[i+1,j] + s[i-1,j]))

    return 1
    #Find the energy of the given array

def model(x):
    #Should be the solution that depends on temperature, etc.
    return 1

def logpost(x):
    return loglikelihood(x) + logprior(x)

def loglikelihood(s, b, energy, new_array):
    #b is the Beta value
    return -b * (energy(new_array(s)) - energy(s))

def logprior(x):
    return 1

def proposal(x): #This is not log
    return np.random.normal() + x

def mcmc(initial, model, prop, post, iterations):
    t = [] #temp array?
    e = [] #energy array?

    return 1

solution = mcmc(1, model, proposal, logpost, 100)
