"""
Basic SIR Model Simulation Example

This script demonstrates a simple SIR epidemic simulation.
"""

import sys
sys.path.insert(0, '/home/user/modsimproj')

import matplotlib.pyplot as plt
from core.base_models import SIRParameters
from models.sir_model import SIRModel


def main():
    # Set up parameters
    params = SIRParameters(
        population=1.0,
        beta=0.5,      # Transmission rate
        gamma=0.1      # Recovery rate
    )

    print(f"SIR Model Parameters:")
    print(f"  Beta (transmission rate): {params.beta}")
    print(f"  Gamma (recovery rate): {params.gamma}")
    print(f"  R0 (basic reproduction number): {params.R0:.2f}")
    print()

    # Create model with initial conditions
    model = SIRModel(
        params=params,
        S0=0.99,  # 99% susceptible
        I0=0.01,  # 1% initially infected
        R0=0.0    # 0% recovered
    )

    # Run simulation
    print("Running simulation...")
    results = model.simulate(t_span=(0, 200))

    # Calculate key metrics
    peak_time, peak_infections = model.calculate_peak_infection(results)
    attack_rate = model.calculate_attack_rate(results)
    herd_immunity = model.calculate_herd_immunity_threshold()

    print(f"\nResults:")
    print(f"  Peak infections: {peak_infections:.1%} at day {peak_time:.1f}")
    print(f"  Attack rate (total infected): {attack_rate:.1%}")
    print(f"  Herd immunity threshold: {herd_immunity:.1%}")

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    model.plot_dynamics(results=results, ax=ax)
    plt.tight_layout()
    plt.savefig('/home/user/modsimproj/results/sir_basic_simulation.png', dpi=300)
    print(f"\nPlot saved to: results/sir_basic_simulation.png")
    plt.show()


if __name__ == "__main__":
    main()
