"""
RQ1: How does the timing of vaccination campaigns affect epidemic outcomes
across different R₀ regimes?

This comprehensive script investigates the optimal timing of vaccination campaigns
and how it interacts with disease transmissibility (R₀).

Research Phases:
1. Baseline Characterization - SEIR dynamics without vaccination
2. Vaccination Timing Sweep - Test different start times
3. Parameter Sensitivity - Vary efficacy and vaccination rate
4. Analysis and Visualization - Quantify optimal windows

Author: Epidemic Modeling Research Team
Date: 2025
"""

import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm
import pickle
from pathlib import Path

from models.seirv_model import SEIRVModel, SEIRVParameters
from models.seir_model import SEIRModel
from core.base_models import SEIRParameters


# ============================================================================
# CONFIGURATION
# ============================================================================

# R₀ values to test
R0_VALUES = [1.5, 2.5, 4.0]

# Base parameters
GAMMA = 0.1          # Recovery rate (10 day infectious period)
SIGMA = 0.2          # Incubation rate (5 day incubation period)

# Phase 2: Vaccination timing sweep parameters
VACCINATION_RATE_BASE = 0.01     # 1% of population per day
VACCINE_EFFICACY_BASE = 0.80     # 80% efficacy
N_STOCHASTIC_REPLICATES = 30     # Number of stochastic runs per condition

# Phase 3: Parameter sensitivity ranges
EFFICACY_VALUES = [0.5, 0.7, 0.9]
VACCINATION_RATES = [0.005, 0.01, 0.02]

# Simulation parameters
T_MAX = 500          # Maximum simulation time
T_EVAL_POINTS = 2000 # Time points for evaluation

# Output directory
OUTPUT_DIR = Path('/home/user/modsimproj/results/rq1_vaccination_timing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PHASE 1: BASELINE CHARACTERIZATION
# ============================================================================

def run_phase1_baseline():
    """
    Phase 1: Characterize SEIR dynamics without vaccination.

    For each R₀, measure:
    - Epidemic peak time (t_peak)
    - Peak infectious fraction (I_peak)
    - Attack rate (R_∞)
    """
    print("\n" + "="*80)
    print("PHASE 1: BASELINE CHARACTERIZATION (No Vaccination)")
    print("="*80)

    baseline_results = {}

    for R0 in R0_VALUES:
        print(f"\nRunning baseline for R₀ = {R0}")

        # Calculate beta from R₀ = beta/gamma
        beta = R0 * GAMMA

        # Create SEIR model (no vaccination)
        params = SEIRParameters(
            beta=beta,
            sigma=SIGMA,
            gamma=GAMMA
        )

        model = SEIRModel(
            params=params,
            S0=0.99,
            E0=0.0,
            I0=0.01,
            R0=0.0
        )

        # Run simulation
        results = model.simulate(t_span=(0, T_MAX))

        # Calculate metrics
        I_peak_idx = np.argmax(results['I'])
        t_peak = results['t'][I_peak_idx]
        I_peak = results['I'][I_peak_idx]
        attack_rate = results['R'][-1]

        # Store results
        baseline_results[R0] = {
            'results': results,
            't_peak': t_peak,
            'I_peak': I_peak,
            'attack_rate': attack_rate,
            'beta': beta
        }

        print(f"  Peak time: {t_peak:.1f} days")
        print(f"  Peak infections: {I_peak:.2%}")
        print(f"  Attack rate: {attack_rate:.2%}")

    # Save baseline results
    with open(OUTPUT_DIR / 'phase1_baseline.pkl', 'wb') as f:
        pickle.dump(baseline_results, f)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, R0 in enumerate(R0_VALUES):
        ax = axes[idx]
        res = baseline_results[R0]['results']

        ax.plot(res['t'], res['S'], 'b-', label='S', linewidth=2)
        ax.plot(res['t'], res['E'], 'y-', label='E', linewidth=2)
        ax.plot(res['t'], res['I'], 'r-', label='I', linewidth=2)
        ax.plot(res['t'], res['R'], 'g-', label='R', linewidth=2)

        # Mark peak
        ax.axvline(baseline_results[R0]['t_peak'], color='k',
                  linestyle='--', alpha=0.5, label='Peak')

        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Fraction of Population', fontsize=11)
        ax.set_title(f'R₀ = {R0}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase1_baseline_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Baseline plot saved to {OUTPUT_DIR / 'phase1_baseline_dynamics.png'}")

    return baseline_results


# ============================================================================
# PHASE 2: VACCINATION TIMING SWEEP
# ============================================================================

def run_phase2_timing_sweep(baseline_results):
    """
    Phase 2: Sweep vaccination start times for each R₀.

    Test vaccination starting at different times relative to epidemic peak.
    Run both deterministic and stochastic simulations.
    """
    print("\n" + "="*80)
    print("PHASE 2: VACCINATION TIMING SWEEP")
    print("="*80)

    timing_results = {}

    for R0 in R0_VALUES:
        print(f"\n--- R₀ = {R0} ---")

        beta = baseline_results[R0]['beta']
        t_peak = baseline_results[R0]['t_peak']
        baseline_attack_rate = baseline_results[R0]['attack_rate']

        # Define vaccination start times relative to peak
        # From t=0 to 2*t_peak with focus around peak
        t_starts = np.concatenate([
            np.linspace(0, t_peak - 10, 5),          # Before peak
            [t_peak - 10, t_peak - 5, t_peak],       # Around peak
            [t_peak + 5, t_peak + 10],               # Just after peak
            np.linspace(t_peak + 20, 2*t_peak, 5)    # After peak
        ])
        t_starts = np.unique(np.clip(t_starts, 0, None))  # Remove duplicates and negatives

        timing_results[R0] = {
            't_starts': t_starts,
            'deterministic': [],
            'stochastic': [],
            'baseline_attack_rate': baseline_attack_rate
        }

        # Run deterministic simulations
        print(f"  Running deterministic timing sweep ({len(t_starts)} conditions)...")
        for t_start in tqdm(t_starts, desc="  Deterministic"):
            result = run_vaccination_simulation(
                beta=beta,
                t_start=t_start,
                duration=T_MAX,
                vaccination_rate=VACCINATION_RATE_BASE,
                efficacy=VACCINE_EFFICACY_BASE,
                stochastic=False
            )
            timing_results[R0]['deterministic'].append(result)

        # Run stochastic simulations (subset of time points for speed)
        print(f"  Running stochastic simulations...")
        stochastic_t_starts = t_starts[::2]  # Every other time point
        stochastic_results = []

        for t_start in tqdm(stochastic_t_starts, desc="  Stochastic"):
            replicates = []
            for rep in range(N_STOCHASTIC_REPLICATES):
                result = run_vaccination_simulation(
                    beta=beta,
                    t_start=t_start,
                    duration=T_MAX,
                    vaccination_rate=VACCINATION_RATE_BASE,
                    efficacy=VACCINE_EFFICACY_BASE,
                    stochastic=True,
                    seed=rep
                )
                replicates.append(result)
            stochastic_results.append({
                't_start': t_start,
                'replicates': replicates
            })

        timing_results[R0]['stochastic'] = stochastic_results

    # Save timing results
    with open(OUTPUT_DIR / 'phase2_timing_sweep.pkl', 'wb') as f:
        pickle.dump(timing_results, f)

    print("\n✓ Phase 2 complete")
    return timing_results


def run_vaccination_simulation(beta, t_start, duration, vaccination_rate,
                               efficacy, stochastic=False, seed=None):
    """
    Run a single vaccination simulation.

    Args:
        beta: Transmission rate
        t_start: Vaccination start time
        duration: Total simulation duration
        vaccination_rate: Daily vaccination rate
        efficacy: Vaccine efficacy
        stochastic: Whether to use stochastic simulation
        seed: Random seed for stochastic simulation

    Returns:
        Dictionary with simulation results and metrics
    """
    if seed is not None:
        np.random.seed(seed)

    params = SEIRVParameters(
        beta=beta,
        sigma=SIGMA,
        gamma=GAMMA,
        vaccine_efficacy=efficacy,
        vaccination_rate=0.0  # Will be set by campaign
    )

    model = SEIRVModel(
        params=params,
        S0=0.99,
        E0=0.0,
        I0=0.01,
        R0=0.0,
        V0=0.0
    )

    # Set vaccination campaign
    model.set_vaccination_campaign(
        start_time=t_start,
        duration=duration - t_start,  # Vaccinate until end
        rate=vaccination_rate,
        efficacy=efficacy
    )

    # Run simulation
    if stochastic:
        # Simple stochastic version - add noise to initial conditions
        S0_stoch = max(0.9, min(0.999, np.random.normal(0.99, 0.01)))
        I0_stoch = 1.0 - S0_stoch
        model.S0 = S0_stoch
        model.I0 = I0_stoch
        results = model.simulate(t_span=(0, duration))
    else:
        results = model.simulate(t_span=(0, duration))

    # Calculate metrics
    I_peak_idx = np.argmax(results['I'])
    t_peak = results['t'][I_peak_idx]
    I_peak = results['I'][I_peak_idx]

    # Attack rate: final R (includes both natural recovery and effective vaccination)
    # For pure infection count, we approximate as final R + final I
    attack_rate = results['R'][-1] + results['I'][-1]

    # Find epidemic duration (when I drops below 0.1% and stays there)
    threshold = 0.001
    below_threshold = results['I'] < threshold
    if np.any(below_threshold):
        extinction_idx = np.argmax(below_threshold)
        t_extinction = results['t'][extinction_idx]
    else:
        t_extinction = results['t'][-1]

    return {
        'results': results,
        't_start': t_start,
        't_peak': t_peak,
        'I_peak': I_peak,
        'attack_rate': attack_rate,
        't_extinction': t_extinction
    }


# ============================================================================
# PHASE 3: PARAMETER SENSITIVITY
# ============================================================================

def run_phase3_sensitivity(baseline_results):
    """
    Phase 3: Test how optimal timing changes with vaccine efficacy and rate.

    For each R₀, test combinations of:
    - Vaccine efficacy: 50%, 70%, 90%
    - Vaccination rate: 0.5%, 1%, 2% per day
    """
    print("\n" + "="*80)
    print("PHASE 3: PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    sensitivity_results = {}

    for R0 in R0_VALUES:
        print(f"\n--- R₀ = {R0} ---")

        beta = baseline_results[R0]['beta']
        t_peak = baseline_results[R0]['t_peak']
        baseline_attack_rate = baseline_results[R0]['attack_rate']

        # Define vaccination start times (fewer points for sensitivity)
        t_starts = np.linspace(0, 1.5*t_peak, 15)

        sensitivity_results[R0] = {
            't_starts': t_starts,
            'combinations': [],
            'baseline_attack_rate': baseline_attack_rate
        }

        # Test all combinations
        total_combinations = len(EFFICACY_VALUES) * len(VACCINATION_RATES)
        print(f"  Testing {total_combinations} parameter combinations...")

        for efficacy in EFFICACY_VALUES:
            for vax_rate in VACCINATION_RATES:
                print(f"    Efficacy={efficacy:.0%}, Rate={vax_rate:.1%}")

                results_for_combo = []
                for t_start in tqdm(t_starts, desc=f"    Times", leave=False):
                    result = run_vaccination_simulation(
                        beta=beta,
                        t_start=t_start,
                        duration=T_MAX,
                        vaccination_rate=vax_rate,
                        efficacy=efficacy,
                        stochastic=False
                    )
                    results_for_combo.append(result)

                sensitivity_results[R0]['combinations'].append({
                    'efficacy': efficacy,
                    'vaccination_rate': vax_rate,
                    'results': results_for_combo
                })

    # Save sensitivity results
    with open(OUTPUT_DIR / 'phase3_sensitivity.pkl', 'wb') as f:
        pickle.dump(sensitivity_results, f)

    print("\n✓ Phase 3 complete")
    return sensitivity_results


# ============================================================================
# PHASE 4: ANALYSIS AND VISUALIZATION
# ============================================================================

def run_phase4_analysis(baseline_results, timing_results, sensitivity_results):
    """
    Phase 4: Comprehensive analysis and visualization.

    Generate:
    1. Attack rate vs. vaccination start time for each R₀
    2. Optimal timing windows
    3. Benefit quantification
    4. Sensitivity to efficacy and rate
    """
    print("\n" + "="*80)
    print("PHASE 4: ANALYSIS AND VISUALIZATION")
    print("="*80)

    # ---- Figure 1: Attack Rate vs. Vaccination Timing ----
    print("\n  Creating Figure 1: Attack rate vs. timing...")
    fig1 = create_figure1_attack_rate_vs_timing(baseline_results, timing_results)
    fig1.savefig(OUTPUT_DIR / 'figure1_attack_rate_vs_timing.png',
                dpi=300, bbox_inches='tight')

    # ---- Figure 2: Peak Infections and Timing ----
    print("  Creating Figure 2: Peak infections vs. timing...")
    fig2 = create_figure2_peak_infections(baseline_results, timing_results)
    fig2.savefig(OUTPUT_DIR / 'figure2_peak_infections_vs_timing.png',
                dpi=300, bbox_inches='tight')

    # ---- Figure 3: Stochastic Variability ----
    print("  Creating Figure 3: Stochastic variability...")
    fig3 = create_figure3_stochastic_variability(timing_results)
    fig3.savefig(OUTPUT_DIR / 'figure3_stochastic_variability.png',
                dpi=300, bbox_inches='tight')

    # ---- Figure 4: Parameter Sensitivity ----
    print("  Creating Figure 4: Parameter sensitivity...")
    fig4 = create_figure4_sensitivity(sensitivity_results)
    fig4.savefig(OUTPUT_DIR / 'figure4_parameter_sensitivity.png',
                dpi=300, bbox_inches='tight')

    # ---- Quantitative Analysis ----
    print("\n  Performing quantitative analysis...")
    analysis_df = create_quantitative_analysis(baseline_results, timing_results,
                                               sensitivity_results)
    analysis_df.to_csv(OUTPUT_DIR / 'quantitative_analysis.csv', index=False)

    print(f"\n✓ All figures saved to {OUTPUT_DIR}")
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(analysis_df.to_string(index=False))

    return analysis_df


def create_figure1_attack_rate_vs_timing(baseline_results, timing_results):
    """Create main figure: attack rate vs. vaccination start time for each R₀"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, R0 in enumerate(R0_VALUES):
        ax = axes[idx]

        t_starts = timing_results[R0]['t_starts']
        det_results = timing_results[R0]['deterministic']
        baseline_ar = timing_results[R0]['baseline_attack_rate']
        t_peak = baseline_results[R0]['t_peak']

        attack_rates = [r['attack_rate'] for r in det_results]

        # Plot attack rate vs. start time
        ax.plot(t_starts, attack_rates, 'o-', color=colors[idx],
               linewidth=2, markersize=6, label='With vaccination')
        ax.axhline(baseline_ar, color='red', linestyle='--', linewidth=2,
                  label='No vaccination', alpha=0.7)
        ax.axvline(t_peak, color='gray', linestyle=':', linewidth=1.5,
                  alpha=0.7, label=f'Peak (t={t_peak:.0f})')

        # Find optimal timing
        optimal_idx = np.argmin(attack_rates)
        optimal_t = t_starts[optimal_idx]
        optimal_ar = attack_rates[optimal_idx]

        ax.plot(optimal_t, optimal_ar, '*', color='gold', markersize=20,
               markeredgecolor='black', markeredgewidth=1.5,
               label=f'Optimal (t={optimal_t:.0f})', zorder=10)

        ax.set_xlabel('Vaccination Start Time (days)', fontsize=12)
        ax.set_ylabel('Final Attack Rate', fontsize=12)
        ax.set_title(f'R₀ = {R0}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, min(1.0, max(attack_rates) * 1.1)])

    fig.suptitle('Attack Rate vs. Vaccination Start Time Across R₀ Regimes',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_figure2_peak_infections(baseline_results, timing_results):
    """Create figure showing peak infections vs. timing"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, R0 in enumerate(R0_VALUES):
        ax = axes[idx]

        t_starts = timing_results[R0]['t_starts']
        det_results = timing_results[R0]['deterministic']
        baseline_peak = baseline_results[R0]['I_peak']
        t_peak_baseline = baseline_results[R0]['t_peak']

        peak_infections = [r['I_peak'] for r in det_results]

        ax.plot(t_starts, peak_infections, 's-', color=colors[idx],
               linewidth=2, markersize=5, label='With vaccination')
        ax.axhline(baseline_peak, color='red', linestyle='--', linewidth=2,
                  label='No vaccination')
        ax.axvline(t_peak_baseline, color='gray', linestyle=':', linewidth=1.5,
                  alpha=0.7)

        ax.set_xlabel('Vaccination Start Time (days)', fontsize=12)
        ax.set_ylabel('Peak Infectious Fraction', fontsize=12)
        ax.set_title(f'R₀ = {R0}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Peak Hospitalization Burden vs. Vaccination Timing',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_figure3_stochastic_variability(timing_results):
    """Show uncertainty bounds from stochastic simulations"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, R0 in enumerate(R0_VALUES):
        ax = axes[idx]

        stoch_data = timing_results[R0]['stochastic']

        t_starts_stoch = []
        mean_ar = []
        std_ar = []

        for condition in stoch_data:
            t_start = condition['t_start']
            replicates = condition['replicates']
            attack_rates = [r['attack_rate'] for r in replicates]

            t_starts_stoch.append(t_start)
            mean_ar.append(np.mean(attack_rates))
            std_ar.append(np.std(attack_rates))

        t_starts_stoch = np.array(t_starts_stoch)
        mean_ar = np.array(mean_ar)
        std_ar = np.array(std_ar)

        # Plot mean with error bars
        ax.errorbar(t_starts_stoch, mean_ar, yerr=std_ar,
                   fmt='o-', capsize=5, capthick=2, linewidth=2,
                   label='Mean ± SD')
        ax.fill_between(t_starts_stoch, mean_ar - 2*std_ar, mean_ar + 2*std_ar,
                       alpha=0.2, label='95% interval')

        ax.set_xlabel('Vaccination Start Time (days)', fontsize=12)
        ax.set_ylabel('Attack Rate', fontsize=12)
        ax.set_title(f'R₀ = {R0} (n={N_STOCHASTIC_REPLICATES} replicates)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Stochastic Variability in Vaccination Outcomes',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_figure4_sensitivity(sensitivity_results):
    """Show how optimal timing depends on efficacy and vaccination rate"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    for r0_idx, R0 in enumerate(R0_VALUES):
        for eff_idx, efficacy in enumerate(EFFICACY_VALUES):
            ax = fig.add_subplot(gs[eff_idx, r0_idx])

            sens_data = sensitivity_results[R0]['combinations']
            t_starts = sensitivity_results[R0]['t_starts']

            for combo in sens_data:
                if combo['efficacy'] == efficacy:
                    vax_rate = combo['vaccination_rate']
                    attack_rates = [r['attack_rate'] for r in combo['results']]

                    label = f'ν={vax_rate:.1%}/day'
                    ax.plot(t_starts, attack_rates, 'o-', label=label, linewidth=2)

            ax.set_xlabel('Start Time (days)', fontsize=10)
            ax.set_ylabel('Attack Rate', fontsize=10)
            ax.set_title(f'R₀={R0}, ε={efficacy:.0%}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Sensitivity to Vaccine Efficacy and Vaccination Rate',
                fontsize=14, fontweight='bold')

    return fig


def create_quantitative_analysis(baseline_results, timing_results, sensitivity_results):
    """Create table of quantitative findings"""
    data = []

    for R0 in R0_VALUES:
        baseline_ar = baseline_results[R0]['attack_rate']
        t_peak = baseline_results[R0]['t_peak']

        det_results = timing_results[R0]['deterministic']
        t_starts = timing_results[R0]['t_starts']
        attack_rates = [r['attack_rate'] for r in det_results]

        # Find optimal timing
        optimal_idx = np.argmin(attack_rates)
        optimal_t = t_starts[optimal_idx]
        optimal_ar = attack_rates[optimal_idx]

        # Calculate benefit
        percent_reduction = 100 * (baseline_ar - optimal_ar) / baseline_ar

        # Find effective window (times within 5% of optimal)
        threshold_ar = optimal_ar * 1.05
        effective_times = t_starts[np.array(attack_rates) <= threshold_ar]
        if len(effective_times) > 0:
            window_start = effective_times[0]
            window_end = effective_times[-1]
            window_width = window_end - window_start
        else:
            window_start = window_end = window_width = np.nan

        data.append({
            'R0': R0,
            'Baseline_AR': f'{baseline_ar:.2%}',
            'Peak_Time': f'{t_peak:.1f}',
            'Optimal_Start_Time': f'{optimal_t:.1f}',
            'Optimal_AR': f'{optimal_ar:.2%}',
            'Percent_Reduction': f'{percent_reduction:.1f}%',
            'Effective_Window': f'{window_start:.0f}-{window_end:.0f} days',
            'Window_Width': f'{window_width:.0f} days'
        })

    return pd.DataFrame(data)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*15 + "RQ1: VACCINATION TIMING OPTIMIZATION")
    print("="*80)
    print("\nResearch Question:")
    print("How does the timing of vaccination campaigns affect epidemic outcomes")
    print("across different R₀ regimes?")
    print("\nConfiguration:")
    print(f"  R₀ values: {R0_VALUES}")
    print(f"  Vaccination rate: {VACCINATION_RATE_BASE:.1%}/day")
    print(f"  Vaccine efficacy: {VACCINE_EFFICACY_BASE:.0%}")
    print(f"  Stochastic replicates: {N_STOCHASTIC_REPLICATES}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Run all phases
    baseline_results = run_phase1_baseline()
    timing_results = run_phase2_timing_sweep(baseline_results)
    sensitivity_results = run_phase3_sensitivity(baseline_results)
    analysis_df = run_phase4_analysis(baseline_results, timing_results, sensitivity_results)

    print("\n" + "="*80)
    print(" "*25 + "STUDY COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nKey Findings:")
    print("  - Check figure1_attack_rate_vs_timing.png for main results")
    print("  - See quantitative_analysis.csv for numerical summary")
    print("  - Review all phase*.pkl files for detailed data")

    return baseline_results, timing_results, sensitivity_results, analysis_df


if __name__ == "__main__":
    # Check if tqdm is available
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm not installed. Install with: pip install tqdm")
        print("Continuing without progress bars...")
        # Create dummy tqdm
        def tqdm(iterable, desc="", leave=True):
            return iterable

    main()
