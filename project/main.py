"""
Main script for project three, Markov Chain Monte Carlo simulation for phy 607

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import tqdm

#Import data

def model(x):
    return 1

def post(x):
    return likelihood(x) * prior(x)

def prior(x):
    return 1

def proposal(x):
    return np.random.normal() + x

def likelihood(m):
    return 1 
    #Just a Gaussian normal for the noise

#Params: 