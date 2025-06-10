
import torch
import numpy as np
import matplotlib.pyplot as plt

class ExperimentSimulator:
    """
    A class for simulating virtual experiments using PyTorch with CUDA acceleration.
    This can be extended for various scientific simulations (physics, biology, etc.).
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the ExperimentSimulator with a specified device.
        Args:
            device (str): The device to use for computations ("cuda" or "cpu").
        """
        self.device = torch.device(device)
        print(f"ExperimentSimulator initialized on device: {self.device}")

    def run_physics_simulation(self, initial_position: float, initial_velocity: float, time_steps: int, dt: float) -> torch.Tensor:
        """
        Simulates a simple 1D physics experiment (e.g., projectile motion under gravity).
        Args:
            initial_position (float): Starting position.
            initial_velocity (float): Starting velocity.
            time_steps (int): Number of simulation steps.
            dt (float): Time step size.
        Returns:
            torch.Tensor: Tensor containing positions at each time step.
        """
        g = torch.tensor(9.81, device=self.device) # Acceleration due to gravity

        positions = torch.zeros(time_steps, device=self.device)
        velocities = torch.zeros(time_steps, device=self.device)

        positions[0] = torch.tensor(initial_position, device=self.device)
        velocities[0] = torch.tensor(initial_velocity, device=self.device)

        for i in range(1, time_steps):
            velocities[i] = velocities[i-1] - g * dt
            positions[i] = positions[i-1] + velocities[i] * dt

        return positions

    def run_biological_simulation(self, initial_population_a: float, initial_population_b: float, growth_rate_a: float, growth_rate_b: float, interaction_rate: float, time_steps: int, dt: float) -> torch.Tensor:
        """
        Simulates a simple Lotka-Volterra predator-prey model.
        Args:
            initial_population_a (float): Initial population of species A (prey).
            initial_population_b (float): Initial population of species B (predator).
            growth_rate_a (float): Growth rate of species A.
            growth_rate_b (float): Death rate of species B.
            interaction_rate (float): Interaction rate between species A and B.
            time_steps (int): Number of simulation steps.
            dt (float): Time step size.
        Returns:
            torch.Tensor: Tensor containing populations of A and B at each time step.
        """
        populations = torch.zeros(time_steps, 2, device=self.device)
        populations[0, 0] = torch.tensor(initial_population_a, device=self.device)
        populations[0, 1] = torch.tensor(initial_population_b, device=self.device)

        for i in range(1, time_steps):
            pop_a = populations[i-1, 0]
            pop_b = populations[i-1, 1]

            # Lotka-Volterra equations
            d_pop_a = growth_rate_a * pop_a - interaction_rate * pop_a * pop_b
            d_pop_b = interaction_rate * pop_a * pop_b - growth_rate_b * pop_b

            populations[i, 0] = pop_a + d_pop_a * dt
            populations[i, 1] = pop_b + d_pop_b * dt

            # Ensure populations don't go below zero
            populations[i, 0] = torch.max(populations[i, 0], torch.tensor(0.0, device=self.device))
            populations[i, 1] = torch.max(populations[i, 1], torch.tensor(0.0, device=self.device))

        return populations

    def generate_synthetic_data(self, num_samples: int, num_features: int, noise_level: float = 0.1) -> torch.Tensor:
        """
        Generates synthetic dataset with configurable parameters.
        Args:
            num_samples (int): Number of data samples.
            num_features (int): Number of features per sample.
            noise_level (float): Standard deviation of added noise.
        Returns:
            torch.Tensor: Generated synthetic data.
        """
        # Generate random data
        data = torch.randn(num_samples, num_features, device=self.device)
        # Add some noise
        noise = torch.randn(num_samples, num_features, device=self.device) * noise_level
        synthetic_data = data + noise
        return synthetic_data

    def visualize_data(self, data: torch.Tensor, title: str = "Simulation Data", labels: list = None):
        """
        Visualizes 1D or 2D data using matplotlib.
        Args:
            data (torch.Tensor): Data to visualize.
            title (str): Title for the plot.
            labels (list): Optional list of labels for different data series (e.g., for biological populations).
        """
        data_np = data.cpu().numpy() # Move to CPU for plotting

        plt.figure(figsize=(10, 6))
        if data_np.ndim == 1:
            plt.plot(data_np)
            plt.xlabel("Time Step" if "positions" in title.lower() else "Sample Index")
            plt.ylabel("Value")
        elif data_np.ndim == 2:
            if data_np.shape[1] == 2:
                if labels:
                    plt.plot(data_np[:, 0], label=labels[0])
                    plt.plot(data_np[:, 1], label=labels[1])
                    plt.legend()
                else:
                    plt.scatter(data_np[:, 0], data_np[:, 1])
                plt.xlabel("Feature 1" if not labels else "Time Step")
                plt.ylabel("Feature 2" if not labels else "Population")
            else:
                print("Visualization currently supports only 1D or 2D data (with 2 features for scatter/line plot).")
                return
        else:
            print("Visualization currently supports only 1D or 2D data.")
            return

        plt.title(title)
        plt.grid(True)
        plt.savefig(f"./{title.replace(' ', '_').lower()}.png")
        plt.close()
        print(f"Visualization saved as {title.replace(' ', '_').lower()}.png")

if __name__ == "__main__":
    # Example Usage
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    simulator = ExperimentSimulator(device=device)

    # Example 1: Run a physics simulation
    print("\n--- Running Physics Simulation ---")
    initial_pos = 0.0
    initial_vel = 10.0
    time_steps = 100
    dt = 0.1
    positions = simulator.run_physics_simulation(initial_pos, initial_vel, time_steps, dt)
    print(f"Simulated positions (first 5): {positions[:5].tolist()}")
    simulator.visualize_data(positions, "Physics Simulation Positions")

    # Example 2: Run a biological simulation (Lotka-Volterra)
    print("\n--- Running Biological Simulation (Lotka-Volterra) ---")
    initial_pop_a = 100.0
    initial_pop_b = 10.0
    growth_a = 0.1
    growth_b = 0.05
    interaction = 0.001
    bio_time_steps = 200
    bio_dt = 0.5
    populations = simulator.run_biological_simulation(initial_pop_a, initial_pop_b, growth_a, growth_b, interaction, bio_time_steps, bio_dt)
    print(f"Simulated populations (first 5): {populations[:5].tolist()}")
    simulator.visualize_data(populations, "Lotka-Volterra Simulation", labels=["Prey Population", "Predator Population"])

    # Example 3: Generate and visualize synthetic data
    print("\n--- Generating Synthetic Data ---")
    num_samples = 1000
    num_features = 2
    synthetic_data = simulator.generate_synthetic_data(num_samples, num_features)
    print(f"Generated synthetic data shape: {synthetic_data.shape}")
    simulator.visualize_data(synthetic_data, "Synthetic Data Scatter Plot")

    num_features_1d = 1
    synthetic_data_1d = simulator.generate_synthetic_data(num_samples, num_features_1d)
    simulator.visualize_data(synthetic_data_1d.squeeze(), "Synthetic 1D Data Plot")


