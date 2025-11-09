#!/usr/bin/env python3
"""
Example script demonstrating the new SIRD model

This shows how to:
1. Create and run the SIRD model programmatically
2. Calculate epidemic metrics
3. Visualize results
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from core.base_models import SIRDParameters
from models.sird_model import SIRDModel


def main():
    print("=" * 60)
    print("SIRD Model Example - Epidemic with Mortality")
    print("=" * 60)

    # Step 1: Create parameters
    print("\n1. Creating model parameters...")
    params = SIRDParameters(
        population=1_000_000,  # 1 million people
        beta=0.5,              # Transmission rate
        gamma=0.1,             # Recovery rate (10 day infectious period)
        mu=0.02                # Mortality rate (2% of infected die)
    )

    print(f"   Population: {params.population:,.0f}")
    print(f"   β (transmission rate): {params.beta}")
    print(f"   γ (recovery rate): {params.gamma}")
    print(f"   μ (mortality rate): {params.mu}")
    print(f"   Basic Reproduction Number (R₀): {params.R0:.2f}")

    # Step 2: Create the model
    print("\n2. Creating SIRD model...")
    model = SIRDModel(
        params=params,
        S0=0.99,   # 99% susceptible
        I0=0.01,   # 1% initially infected
        R0=0.0,    # 0% recovered
        D0=0.0     # 0% dead
    )
    print(f"   Model created with compartments: {model.state_names}")

    # Step 3: Run simulation
    print("\n3. Running simulation for 200 days...")
    results = model.simulate(t_span=(0, 200))
    print(f"   Simulation complete! {len(results['t'])} time points.")

    # Step 4: Calculate metrics
    print("\n4. Calculating epidemic metrics...")
    summary = model.get_epidemic_summary(results)

    print(f"\n   Basic Reproduction Number (R₀): {summary['R0']:.2f}")
    print(f"   Herd Immunity Threshold: {summary['herd_immunity_threshold']:.1%}")
    print(f"   Peak Infection Time: Day {summary['peak_infection_time']:.1f}")
    print(f"   Peak Infection Level: {summary['peak_infection_fraction']:.2%}")
    print(f"   Attack Rate: {summary['attack_rate']:.1%}")
    print(f"   Case Fatality Rate (CFR): {summary['case_fatality_rate']:.2%}")
    print(f"   Total Deaths: {summary['total_deaths_count']:,.0f} people")
    print(f"   Final Susceptible: {summary['final_susceptible']:.1%}")
    print(f"   Final Recovered: {summary['final_recovered']:.1%}")

    # Step 5: Visualize
    print("\n5. Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot all compartments
    model.plot_dynamics(results=results, ax=ax1)
    ax1.set_title('SIRD Model: All Compartments Over Time')

    # Plot just Deaths
    ax2.plot(results['t'], results['D'] * params.population,
             color='black', linewidth=2)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Number of Deaths')
    ax2.set_title('Cumulative Deaths Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=summary['total_deaths_count'],
                color='red', linestyle='--', alpha=0.5,
                label=f'Final: {summary["total_deaths_count"]:,.0f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/sird_example.png', dpi=300, bbox_inches='tight')
    print("   Plot saved to: results/sird_example.png")

    print("\n" + "=" * 60)
    print("✓ Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
