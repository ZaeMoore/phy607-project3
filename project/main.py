"""
Main script for project three, Markov Chain Monte Carlo simulation for phy 607

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import tqdm

#Import data

def data(N):
    latticeSpins = np.zeros(shape=(N, N))

    for i in latticeSpins:
        for j in i:
            latticeSpins[i,j] = np.random.choice([-1, 1])

    return latticeSpins

def new_array(s, N):
    #s is the spins array
    randomRow = np.random.choice(range(0, N-1))
    randomColumn = np.random.choice(range(0, N-1))
    s[randomRow, randomColumn] *= -1
    return s
    #Given the old array, pick a random spin entry to flip and return the new array

def energy(s):
    return 1
    #Find the energy of the given array

def model(x):
    return 1

def logpost(x):
    return loglikelihood(x) + logprior(x)

def loglikelihood(s, b):
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
