"""
SEIR Model with Interventions Example

This script demonstrates how to simulate interventions like lockdowns
and social distancing in an SEIR epidemic model.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
from core.base_models import SEIRParameters
from models.mixin_models import SEIRWithInterventions
from interventions.strategies import lockdown, social_distancing
from analysis.metrics import compare_interventions


def main():
    # Set up parameters
    params = SEIRParameters(
        population=1.0,
        beta=0.6,      # Transmission rate
        sigma=0.2,     # Incubation rate (1/5 day incubation)
        gamma=0.1      # Recovery rate (10 day infectious period)
    )

    print(f"SEIR Model Parameters:")
    print(f"  Beta (transmission rate): {params.beta}")
    print(f"  Sigma (incubation rate): {params.sigma}")
    print(f"  Gamma (recovery rate): {params.gamma}")
    print(f"  R0: {params.R0:.2f}")
    print(f"  Incubation period: {params.incubation_period:.1f} days")
    print()

    # Scenario 1: No intervention
    print("Scenario 1: No intervention")
    model_no_intervention = SEIRWithInterventions(
        params=params,
        S0=0.99, E0=0.0, I0=0.01, R0=0.0
    )
    results_no_intervention = model_no_intervention.simulate(t_span=(0, 300))

    # Scenario 2: Early lockdown
    print("Scenario 2: Early lockdown (day 30-60, 70% effective)")
    model_early_lockdown = SEIRWithInterventions(
        params=params,
        S0=0.99, E0=0.0, I0=0.01, R0=0.0
    )
    model_early_lockdown.add_intervention(
        start_time=30,
        duration=30,
        effectiveness=0.7,
        intervention_type="reduction"
    )
    results_early_lockdown = model_early_lockdown.simulate(t_span=(0, 300))

    # Scenario 3: Late lockdown
    print("Scenario 3: Late lockdown (day 60-90, 70% effective)")
    model_late_lockdown = SEIRWithInterventions(
        params=params,
        S0=0.99, E0=0.0, I0=0.01, R0=0.0
    )
    model_late_lockdown.add_intervention(
        start_time=60,
        duration=30,
        effectiveness=0.7,
        intervention_type="reduction"
    )
    results_late_lockdown = model_late_lockdown.simulate(t_span=(0, 300))

    # Scenario 4: Sustained social distancing
    print("Scenario 4: Sustained social distancing (day 20-150, 40% effective)")
    model_social_distancing = SEIRWithInterventions(
        params=params,
        S0=0.99, E0=0.0, I0=0.01, R0=0.0
    )
    model_social_distancing.add_intervention(
        start_time=20,
        duration=130,
        effectiveness=0.4,
        intervention_type="reduction"
    )
    results_social_distancing = model_social_distancing.simulate(t_span=(0, 300))

    # Compare interventions
    results_list = [
        results_no_intervention,
        results_early_lockdown,
        results_late_lockdown,
        results_social_distancing
    ]
    labels = [
        'No Intervention',
        'Early Lockdown',
        'Late Lockdown',
        'Sustained Social Distancing'
    ]

    comparison = compare_interventions(results_list, labels)

    print("\nComparison of Interventions:")
    print("-" * 80)
    for i, label in enumerate(labels):
        print(f"{label}:")
        print(f"  Peak infections: {comparison['peak_infections'][i]:.2%} at day {comparison['peak_times'][i]:.1f}")
        print(f"  Total infections: {comparison['total_infections'][i]:.2%}")
        print(f"  Epidemic duration: {comparison['durations'][i]:.1f} days")
        print()

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Infectious compartment
    for results, label in zip(results_list, labels):
        axes[0, 0].plot(results['t'], results['I'], label=label, linewidth=2)
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious (fraction)')
    axes[0, 0].set_title('Infectious Compartment Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot Susceptible compartment
    for results, label in zip(results_list, labels):
        axes[0, 1].plot(results['t'], results['S'], label=label, linewidth=2)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Susceptible (fraction)')
    axes[0, 1].set_title('Susceptible Compartment Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot Exposed compartment
    for results, label in zip(results_list, labels):
        axes[1, 0].plot(results['t'], results['E'], label=label, linewidth=2)
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Exposed (fraction)')
    axes[1, 0].set_title('Exposed Compartment Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot Recovered compartment
    for results, label in zip(results_list, labels):
        axes[1, 1].plot(results['t'], results['R'], label=label, linewidth=2)
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Recovered (fraction)')
    axes[1, 1].set_title('Recovered Compartment Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/modsimproj/results/seir_interventions_comparison.png', dpi=300)
    print(f"Plot saved to: results/seir_interventions_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
