"""
Model Comparison Example

This script compares SIR, SEIR, and SIRS models.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
from core.base_models import SIRParameters, SEIRParameters, SIRSParameters
from models.sir_model import SIRModel
from models.seir_model import SEIRModel
from models.sirs_model import SIRSModel


def main():
    # Common parameters
    beta = 0.5
    gamma = 0.1

    print("Comparing SIR, SEIR, and SIRS Models")
    print("=" * 60)
    print()

    # SIR Model
    print("1. SIR Model (Classic)")
    sir_params = SIRParameters(beta=beta, gamma=gamma)
    sir_model = SIRModel(params=sir_params, S0=0.99, I0=0.01, R0=0.0)
    sir_results = sir_model.simulate(t_span=(0, 200))
    peak_t, peak_i = sir_model.calculate_peak_infection()
    print(f"   R0: {sir_params.R0:.2f}")
    print(f"   Peak infections: {peak_i:.2%} at day {peak_t:.1f}")
    print()

    # SEIR Model
    print("2. SEIR Model (with latent period)")
    seir_params = SEIRParameters(beta=beta, sigma=0.2, gamma=gamma)
    seir_model = SEIRModel(params=seir_params, S0=0.99, E0=0.0, I0=0.01, R0=0.0)
    seir_results = seir_model.simulate(t_span=(0, 200))
    peak_t, peak_i = seir_model.calculate_peak_infection()
    print(f"   R0: {seir_params.R0:.2f}")
    print(f"   Incubation period: {seir_params.incubation_period:.1f} days")
    print(f"   Peak infections: {peak_i:.2%} at day {peak_t:.1f}")
    print()

    # SIRS Model
    print("3. SIRS Model (with waning immunity)")
    sirs_params = SIRSParameters(beta=beta, gamma=gamma, omega=0.01)
    sirs_model = SIRSModel(params=sirs_params, S0=0.99, I0=0.01, R0=0.0)
    sirs_results = sirs_model.simulate(t_span=(0, 500))  # Longer time span for oscillations
    endemic = sirs_model.calculate_endemic_equilibrium()
    print(f"   R0: {sirs_params.R0:.2f}")
    print(f"   Immunity waning rate: {sirs_params.omega}")
    print(f"   Endemic equilibrium: S={endemic['S']:.2%}, I={endemic['I']:.2%}, R={endemic['R']:.2%}")
    print()

    # Create comparison plots
    fig = plt.figure(figsize=(15, 10))

    # SIR Model
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(sir_results['t'], sir_results['S'], 'b-', label='S', linewidth=2)
    ax1.plot(sir_results['t'], sir_results['I'], 'r-', label='I', linewidth=2)
    ax1.plot(sir_results['t'], sir_results['R'], 'g-', label='R', linewidth=2)
    ax1.set_title('SIR Model Dynamics')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Fraction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SEIR Model
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(seir_results['t'], seir_results['S'], 'b-', label='S', linewidth=2)
    ax2.plot(seir_results['t'], seir_results['E'], 'y-', label='E', linewidth=2)
    ax2.plot(seir_results['t'], seir_results['I'], 'r-', label='I', linewidth=2)
    ax2.plot(seir_results['t'], seir_results['R'], 'g-', label='R', linewidth=2)
    ax2.set_title('SEIR Model Dynamics')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Fraction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # SIRS Model
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(sirs_results['t'], sirs_results['S'], 'b-', label='S', linewidth=2)
    ax3.plot(sirs_results['t'], sirs_results['I'], 'r-', label='I', linewidth=2)
    ax3.plot(sirs_results['t'], sirs_results['R'], 'g-', label='R', linewidth=2)
    ax3.set_title('SIRS Model Dynamics (with oscillations)')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Fraction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Compare Infectious compartments
    ax4 = plt.subplot(3, 1, 2)
    ax4.plot(sir_results['t'], sir_results['I'], label='SIR', linewidth=2)
    ax4.plot(seir_results['t'], seir_results['I'], label='SEIR', linewidth=2)
    ax4.plot(sirs_results['t'], sirs_results['I'], label='SIRS', linewidth=2)
    ax4.set_title('Infectious Compartment Comparison')
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Infectious (fraction)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Phase portraits
    ax5 = plt.subplot(3, 3, 7)
    ax5.plot(sir_results['S'], sir_results['I'], 'b-', linewidth=2)
    ax5.plot(sir_results['S'][0], sir_results['I'][0], 'go', markersize=8, label='Start')
    ax5.plot(sir_results['S'][-1], sir_results['I'][-1], 'ro', markersize=8, label='End')
    ax5.set_title('SIR Phase Portrait')
    ax5.set_xlabel('Susceptible')
    ax5.set_ylabel('Infectious')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 8)
    ax6.plot(seir_results['S'], seir_results['I'], 'b-', linewidth=2)
    ax6.plot(seir_results['S'][0], seir_results['I'][0], 'go', markersize=8, label='Start')
    ax6.plot(seir_results['S'][-1], seir_results['I'][-1], 'ro', markersize=8, label='End')
    ax6.set_title('SEIR Phase Portrait')
    ax6.set_xlabel('Susceptible')
    ax6.set_ylabel('Infectious')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 9)
    ax7.plot(sirs_results['S'], sirs_results['I'], 'b-', linewidth=2)
    ax7.plot(sirs_results['S'][0], sirs_results['I'][0], 'go', markersize=8, label='Start')
    ax7.plot(endemic['S'], endemic['I'], 'r*', markersize=15, label='Endemic Equilibrium')
    ax7.set_title('SIRS Phase Portrait')
    ax7.set_xlabel('Susceptible')
    ax7.set_ylabel('Infectious')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/modsimproj/results/model_comparison.png', dpi=300)
    print("Plot saved to: results/model_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
