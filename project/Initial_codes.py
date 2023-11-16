import numpy as np
import matplotlib.pyplot as plt

# Particle Class
class Particle:
    def __init__(self):
        self.mass = np.random.rand()          # Randomly choose mass between 0 and 1.
        self.charge = np.random.uniform(-1, 1) # Randomly choose charge between -1 and 1.
        u = np.random.rand()
        self.position = -np.log(1 - u)        # Using the inverse transform sampling method.
        self.velocity = 0                     # Initial velocity is set to 0.

    def acceleration(self, E):
        return self.charge * E / self.mass    # Acceleration due to electric field.

# Simulation Class
class Simulation:
    def __init__(self, num_particles=1000, E=1, dt=0.1, T=1000):
        self.particles = [Particle() for _ in range(num_particles)]  # Initialize list of particles.
        self.E = E                                                   # Electric field strength.
        self.dt = dt                                                 # Time step.
        self.time_steps = int(T/dt)                                  # Number of time steps.
        self.positions = np.zeros((num_particles, self.time_steps))  # To store positions.
        self.velocities = np.zeros((num_particles, self.time_steps)) # To store velocities.

    def rk4_step(self, particle):
        # RK4 method to update position and velocity.
        k1_v = particle.acceleration(self.E) * self.dt
        k1_x = particle.velocity * self.dt

        k2_v = (particle.acceleration(self.E) + k1_v/2) * self.dt
        k2_x = (particle.velocity + k1_v/2) * self.dt

        k3_v = (particle.acceleration(self.E) + k2_v/2) * self.dt
        k3_x = (particle.velocity + k2_v/2) * self.dt

        k4_v = (particle.acceleration(self.E) + k3_v) * self.dt
        k4_x = (particle.velocity + k3_v) * self.dt

        dx = (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        dv = (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

        return dx, dv

    def evolve(self):
        for i in range(self.time_steps-1):
            for idx, particle in enumerate(self.particles):
                dx, dv = self.rk4_step(particle)
                particle.position += dx
                particle.velocity += dv
                self.positions[idx, i+1] = particle.position
                self.velocities[idx, i+1] = particle.velocity

    def plot_distribution(self):
        def expected_pdf(x):
            return np.exp(-x)

        final_positions = self.positions[:, -1]
        plt.figure(figsize=(10, 6))
        plt.hist(final_positions, bins=50, density=True, alpha=0.7, label='Observed Distribution')
        x_vals = np.linspace(0, np.max(final_positions), 400)
        y_vals = [expected_pdf(x) for x in x_vals]
        plt.plot(x_vals, y_vals, 'r-', label='Expected PDF')
        plt.title("Observed Particle Distribution vs. Expected PDF")
        plt.xlabel("Position")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

# Main Execution
if __name__ == "__main__":
    sim = Simulation()
    sim.evolve()
    sim.plot_distribution()
