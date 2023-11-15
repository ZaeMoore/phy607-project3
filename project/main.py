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

def dH(s):
    return new_array(s)
    #Find new array s', calculate the difference between H(s) and H(s')
    #This will be used to determine if the spin flips

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