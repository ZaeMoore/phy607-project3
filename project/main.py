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

def logpost(x):
    return loglikelihood(x) * logprior(x)

def loglikelihood(m):
    return 1 
    #Just a Gaussian normal for the noise

def logprior(x):
    return 1

def proposal(x): #This is not log
    return np.random.normal() + x


#Params: 