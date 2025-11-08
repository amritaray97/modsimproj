"""
Stochastic SIR Simulation Example

This script demonstrates stochastic epidemic simulations and compares
them with deterministic models.
"""

import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')
from typing import Optional


import matplotlib.pyplot as plt
import numpy as np
from modsimproj.core.base_models import SIRParameters
from modsimproj.models.mixin_models import StochasticSIR


def main():
    params = SIRParameters(
        population=1.0,
        beta=0.5,
        gamma=0.1
    )

    print("Stochastic SIR Model Simulation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Beta: {params.beta}")
    print(f"  Gamma: {params.gamma}")
    print(f"  R0: {params.R0:.2f}")
    print()

    model = StochasticSIR(
        params=params,
        S0=0.99,
        I0=0.01,
        R0=0.0
    )

    print("Running deterministic simulation...")
    det_results = model.simulate(t_span=(0, 200))

    # multiple stochastic simulations
    n_simulations = 10
    print(f"Running {n_simulations} stochastic simulations...")
    stoch_results = []

    initial_conditions = model.get_initial_conditions()

    for i in range(n_simulations):
        np.random.seed(i)  # For reproducibility
        results = model.simulate_stochastic(
            initial_conditions=initial_conditions,
            t_span=(0, 200),
            dt=0.1,
            noise_scale=0.02
        )
        stoch_results.append(results)
        print(f"  Simulation {i+1}/{n_simulations} complete")

    print("\nSimulations complete!")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Susceptible
    ax = axes[0, 0]
    ax.plot(det_results['t'], det_results['S'], 'b-', linewidth=3,
           label='Deterministic', zorder=10)
    for i, results in enumerate(stoch_results):
        ax.plot(results['t'], results['S'], 'b-', alpha=0.3, linewidth=1)
    ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label='Stochastic')
    ax.set_title('Susceptible Compartment')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Infectious
    ax = axes[0, 1]
    ax.plot(det_results['t'], det_results['I'], 'r-', linewidth=3,
           label='Deterministic', zorder=10)
    for i, results in enumerate(stoch_results):
        ax.plot(results['t'], results['I'], 'r-', alpha=0.3, linewidth=1)
    ax.plot([], [], 'r-', alpha=0.3, linewidth=1, label='Stochastic')
    ax.set_title('Infectious Compartment')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recovered
    ax = axes[1, 0]
    ax.plot(det_results['t'], det_results['R'], 'g-', linewidth=3,
           label='Deterministic', zorder=10)
    for i, results in enumerate(stoch_results):
        ax.plot(results['t'], results['R'], 'g-', alpha=0.3, linewidth=1)
    ax.plot([], [], 'g-', alpha=0.3, linewidth=1, label='Stochastic')
    ax.set_title('Recovered Compartment')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribution of peak infections
    ax = axes[1, 1]
    peak_infections = []
    peak_times = []
    for results in stoch_results:
        peak_idx = np.argmax(results['I'])
        peak_infections.append(results['I'][peak_idx])
        peak_times.append(results['t'][peak_idx])

    det_peak_idx = np.argmax(det_results['I'])
    det_peak_infection = det_results['I'][det_peak_idx]
    det_peak_time = det_results['t'][det_peak_idx]

    ax.scatter(peak_times, peak_infections, alpha=0.6, s=100, label='Stochastic')
    ax.scatter([det_peak_time], [det_peak_infection], color='red', s=200,
              marker='*', label='Deterministic', zorder=10)
    ax.set_title('Peak Infection Distribution')
    ax.set_xlabel('Time of Peak (days)')
    ax.set_ylabel('Peak Infection (fraction)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/modsimproj/results/stochastic_comparison.png', dpi=300)
    print("\nPlot saved to: results/stochastic_comparison.png")

    # statistics
    print("\nStatistics:")
    print(f"Deterministic peak: {det_peak_infection:.2%} at day {det_peak_time:.1f}")
    print(f"Stochastic peaks:")
    print(f"  Mean: {np.mean(peak_infections):.2%} at day {np.mean(peak_times):.1f}")
    print(f"  Std dev: {np.std(peak_infections):.2%} (infection), {np.std(peak_times):.1f} (time)")
    print(f"  Range: {np.min(peak_infections):.2%} - {np.max(peak_infections):.2%}")

    plt.show()


if __name__ == "__main__":
    main()
