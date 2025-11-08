"""
RQ1: Vaccination Timing - QUICK VERSION

This is a faster version of the full research script for quick testing
and demonstration purposes. It uses reduced parameters:
- Fewer R₀ values
- Fewer timing points
- Fewer stochastic replicates
- Fewer sensitivity combinations

For the full analysis, run: rq1_vaccination_timing.py
"""

import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.seirv_model import SEIRVModel, SEIRVParameters
from models.seir_model import SEIRModel
from core.base_models import SEIRParameters


# ============================================================================
# QUICK CONFIGURATION (Reduced for speed)
# ============================================================================

R0_VALUES = [1.5, 4.0]  # Just two extremes
GAMMA = 0.1
SIGMA = 0.2
VACCINATION_RATE_BASE = 0.01
VACCINE_EFFICACY_BASE = 0.80
N_STOCHASTIC_REPLICATES = 5  # Reduced from 30
T_MAX = 300  # Reduced from 500

OUTPUT_DIR = Path('/home/user/modsimproj/results/rq1_quick')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_quick_analysis():
    """Run quick version of the analysis"""
    print("\n" + "="*70)
    print(" "*15 + "RQ1: VACCINATION TIMING (QUICK VERSION)")
    print("="*70)
    print("\nThis is a faster version with reduced parameters.")
    print("For full analysis, run: rq1_vaccination_timing.py\n")

    results = {}

    for R0 in R0_VALUES:
        print(f"\n{'='*70}")
        print(f"R₀ = {R0}")
        print('='*70)

        beta = R0 * GAMMA

        # 1. Baseline (no vaccination)
        print("  Running baseline simulation...")
        params_base = SEIRParameters(beta=beta, sigma=SIGMA, gamma=GAMMA)
        model_base = SEIRModel(params=params_base, S0=0.99, E0=0.0, I0=0.01, R0=0.0)
        baseline = model_base.simulate(t_span=(0, T_MAX))

        t_peak_idx = np.argmax(baseline['I'])
        t_peak = baseline['t'][t_peak_idx]
        I_peak_baseline = baseline['I'][t_peak_idx]
        baseline_ar = baseline['R'][-1]

        print(f"    Baseline peak: {I_peak_baseline:.2%} at t={t_peak:.1f} days")
        print(f"    Baseline attack rate: {baseline_ar:.2%}")

        # 2. Vaccination timing sweep (fewer points)
        print(f"  Testing vaccination timing...")
        t_starts = np.linspace(0, 1.5*t_peak, 10)  # Just 10 points

        timing_results = []
        for t_start in t_starts:
            params_vax = SEIRVParameters(
                beta=beta, sigma=SIGMA, gamma=GAMMA,
                vaccine_efficacy=VACCINE_EFFICACY_BASE
            )
            model_vax = SEIRVModel(
                params=params_vax,
                S0=0.99, E0=0.0, I0=0.01, R0=0.0, V0=0.0
            )
            model_vax.set_vaccination_campaign(
                start_time=t_start,
                duration=T_MAX - t_start,
                rate=VACCINATION_RATE_BASE
            )
            res = model_vax.simulate(t_span=(0, T_MAX))

            I_peak_idx = np.argmax(res['I'])
            timing_results.append({
                't_start': t_start,
                'attack_rate': res['R'][-1] + res['I'][-1],
                'I_peak': res['I'][I_peak_idx]
            })

        # Find optimal
        attack_rates = [r['attack_rate'] for r in timing_results]
        optimal_idx = np.argmin(attack_rates)
        optimal_t = t_starts[optimal_idx]
        optimal_ar = attack_rates[optimal_idx]
        reduction = 100 * (baseline_ar - optimal_ar) / baseline_ar

        print(f"    Optimal start time: {optimal_t:.1f} days")
        print(f"    Optimal attack rate: {optimal_ar:.2%}")
        print(f"    Reduction: {reduction:.1f}%")

        results[R0] = {
            'baseline': baseline,
            't_peak': t_peak,
            'baseline_ar': baseline_ar,
            't_starts': t_starts,
            'timing_results': timing_results,
            'optimal_t': optimal_t,
            'optimal_ar': optimal_ar,
            'reduction': reduction
        }

    # Visualization
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Baseline dynamics
    for idx, R0 in enumerate(R0_VALUES):
        ax = axes[0, idx]
        res = results[R0]['baseline']

        ax.plot(res['t'], res['S'], 'b-', label='S', linewidth=2)
        ax.plot(res['t'], res['E'], 'y-', label='E', linewidth=2)
        ax.plot(res['t'], res['I'], 'r-', label='I', linewidth=2)
        ax.plot(res['t'], res['R'], 'g-', label='R', linewidth=2)
        ax.axvline(results[R0]['t_peak'], color='k', linestyle='--',
                  alpha=0.5, label='Peak')

        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Fraction', fontsize=11)
        ax.set_title(f'Baseline Dynamics (R₀={R0})', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Plot 2: Attack rate vs. timing
    for idx, R0 in enumerate(R0_VALUES):
        ax = axes[1, idx]

        t_starts = results[R0]['t_starts']
        attack_rates = [r['attack_rate'] for r in results[R0]['timing_results']]
        baseline_ar = results[R0]['baseline_ar']
        t_peak = results[R0]['t_peak']
        optimal_t = results[R0]['optimal_t']
        optimal_ar = results[R0]['optimal_ar']

        ax.plot(t_starts, attack_rates, 'o-', linewidth=2, markersize=8,
               color='#1f77b4', label='With vaccination')
        ax.axhline(baseline_ar, color='red', linestyle='--', linewidth=2,
                  label='No vaccination')
        ax.axvline(t_peak, color='gray', linestyle=':', alpha=0.7,
                  label=f'Peak (t={t_peak:.0f})')
        ax.plot(optimal_t, optimal_ar, '*', color='gold', markersize=20,
               markeredgecolor='black', markeredgewidth=1.5,
               label=f'Optimal (t={optimal_t:.0f})', zorder=10)

        ax.set_xlabel('Vaccination Start Time (days)', fontsize=11)
        ax.set_ylabel('Final Attack Rate', fontsize=11)
        ax.set_title(f'Attack Rate vs. Timing (R₀={R0})', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quick_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {OUTPUT_DIR / 'quick_analysis.png'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for R0 in R0_VALUES:
        print(f"\nR₀ = {R0}:")
        print(f"  Baseline attack rate: {results[R0]['baseline_ar']:.2%}")
        print(f"  Optimal start time: {results[R0]['optimal_t']:.1f} days")
        print(f"  Optimal attack rate: {results[R0]['optimal_ar']:.2%}")
        print(f"  Percent reduction: {results[R0]['reduction']:.1f}%")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    if results[4.0]['optimal_t'] < results[1.5]['optimal_t']:
        print("For high R₀, early vaccination is MORE critical than for low R₀.")
        print("High transmissibility leaves less room for delayed intervention.")
    else:
        print("Timing flexibility depends on epidemic speed and R₀.")

    print("\n" + "="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
    print("\nFor comprehensive analysis with:")
    print("  - More R₀ values")
    print("  - Stochastic simulations")
    print("  - Parameter sensitivity")
    print("  - Detailed metrics")
    print("\nRun: python experiments/rq1_vaccination_timing.py")

    plt.show()

    return results


if __name__ == "__main__":
    results = run_quick_analysis()
