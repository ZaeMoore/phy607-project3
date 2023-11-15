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
    return 1
    #Return an NxN array of spins

def new_array(s):
    return 1
    #Given the old array, pick a random spin entry to flip and return the new array

def energy(s):
    return 1
    #Find the energy of the given array

def model(x):
    return 1

def logpost(x):
    return loglikelihood(x) + logprior(x)

def loglikelihood(s):
    b = 1
    return -b * (energy(new_array(s)) - energy(s))

def logprior(x):
    return 1

def proposal(x): #This is not log
    return np.random.normal() + x

def mcmc(initial, model, prop, post, iterations):
    t = [] #temp array?
    e = [] #energy array?
    return 1
