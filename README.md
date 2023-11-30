A Markov Chain Monte Carlo simulation of the 2D Ising Model.


This script will produce a variety of data, including energies, magnetizations, lattice states, and more.

To run, simply install the package with "pip install ." and enter "run-chain" in the terminal. The simulation will take around 8 minutes to run and an additional two and half minutes for multile-chain analysis/Gelman-Rubin statistics. 


The tests directory holds valuable guidance as well, including example outputs and jupyter notebooks.

There are 3 directories inside the tests directory, each one indicated tests that were run with different initial conditions. neg1_start_tests contains examples where the lattice was initialized with entirely -1 spins. pos1_start_tests contains examples where the lattice was initialized with entirely +1 spins. random_start_tests contains examples where the lattice was initialized with random spins.

Each test directory contains a jupyter notebook and output pngs
