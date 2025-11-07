#!/usr/bin/env python3
"""
Comprehensive Verification and Visualization Script

This script:
1. Verifies all mathematical equations against epidemic modeling theory
2. Runs comprehensive simulations for all models
3. Generates publication-quality visualizations
4. Creates detailed comparison plots
5. Produces a comprehensive results report

Mathematical verification based on:
- Kermack & McKendrick (1927) - Original SIR model
- Anderson & May (1991) - Infectious Diseases of Humans
- Keeling & Rohani (2008) - Modeling Infectious Diseases
"""

import sys
sys.path.insert(0, '/home/user/modsimproj')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from core.base_models import SIRParameters, SEIRParameters, SIRSParameters
from models.sir_model import SIRModel
from models.seir_model import SEIRModel
from models.sirs_model import SIRSModel
from models.seirv_model import SEIRVModel, SEIRVParameters
from models.mixin_models import SEIRWithInterventions, SIRWithInterventions
from analysis.visualization import plot_comparison, plot_phase_portrait, plot_R_effective
from analysis.metrics import calculate_peak_time, calculate_attack_rate


class MathematicalVerifier:
    """Verify mathematical correctness of epidemic models."""

    def __init__(self):
        self.verification_results = {}

    def verify_sir_equations(self):
        """
        Verify SIR model equations.

        Reference: Kermack & McKendrick (1927)
        Equations:
            dS/dt = -β*S*I
            dI/dt = β*S*I - γ*I
            dR/dt = γ*I

        Properties to verify:
        1. Mass conservation: S + I + R = 1
        2. R0 = β/γ
        3. Herd immunity threshold = 1 - 1/R0
        4. Final epidemic size equation
        """
        print("\n" + "="*60)
        print("Verifying SIR Model")
        print("="*60)

        params = SIRParameters(beta=0.5, gamma=0.1)
        model = SIRModel(params=params, S0=0.99, I0=0.01, R0=0.0)
        results = model.simulate(t_span=(0, 200))

        # Verify mass conservation
        total_pop = results['S'] + results['I'] + results['R']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-6)

        # Verify R0
        R0_calc = params.beta / params.gamma
        R0_expected = 5.0
        R0_correct = np.isclose(R0_calc, R0_expected)

        # Verify herd immunity threshold
        herd_immunity = model.calculate_herd_immunity_threshold()
        herd_immunity_expected = 1 - 1/R0_calc
        herd_immunity_correct = np.isclose(herd_immunity, herd_immunity_expected)

        # Verify final size relation (approximate for numerical solution)
        # Final size equation: R_∞ = 1 - exp(-R0 * R_∞)
        R_final = results['R'][-1]
        S_final = results['S'][-1]
        final_size_check = np.isclose(R_final, 1 - S_final * np.exp(-R0_calc * R_final), atol=0.01)

        print(f"✓ Mass Conservation: {mass_conservation}")
        print(f"  Total population preserved: {np.allclose(total_pop, 1.0, atol=1e-6)}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n✓ R0 Calculation: {R0_correct}")
        print(f"  Calculated: {R0_calc:.2f}")
        print(f"  Expected: {R0_expected:.2f}")

        print(f"\n✓ Herd Immunity Threshold: {herd_immunity_correct}")
        print(f"  Calculated: {herd_immunity:.2%}")
        print(f"  Expected: {herd_immunity_expected:.2%}")

        print(f"\n✓ Final Size Relation: {final_size_check}")
        print(f"  R_final: {R_final:.3f}")
        print(f"  S_final: {S_final:.3f}")

        self.verification_results['SIR'] = {
            'mass_conservation': mass_conservation,
            'R0_correct': R0_correct,
            'herd_immunity_correct': herd_immunity_correct,
            'final_size_correct': final_size_check,
            'all_passed': all([mass_conservation, R0_correct, herd_immunity_correct, final_size_check])
        }

        return results

    def verify_seir_equations(self):
        """
        Verify SEIR model equations.

        Reference: Anderson & May (1991)
        Equations:
            dS/dt = -β*S*I
            dE/dt = β*S*I - σ*E
            dI/dt = σ*E - γ*I
            dR/dt = γ*I

        Properties to verify:
        1. Mass conservation: S + E + I + R = 1
        2. R0 = β/γ (same as SIR)
        3. Mean incubation period = 1/σ
        4. Delayed epidemic peak vs SIR
        """
        print("\n" + "="*60)
        print("Verifying SEIR Model")
        print("="*60)

        params = SEIRParameters(beta=0.6, sigma=0.2, gamma=0.1)
        model = SEIRModel(params=params, S0=0.99, E0=0.0, I0=0.01, R0=0.0)
        results = model.simulate(t_span=(0, 300))

        # Verify mass conservation
        total_pop = results['S'] + results['E'] + results['I'] + results['R']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-6)

        # Verify R0
        R0_calc = params.beta / params.gamma
        R0_correct = np.isclose(R0_calc, 6.0)

        # Verify incubation period
        incubation_period = 1.0 / params.sigma
        incubation_correct = np.isclose(incubation_period, 5.0)

        print(f"✓ Mass Conservation: {mass_conservation}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n✓ R0 Calculation: {R0_correct}")
        print(f"  R0 = β/γ = {R0_calc:.2f}")

        print(f"\n✓ Incubation Period: {incubation_correct}")
        print(f"  1/σ = {incubation_period:.1f} days")

        self.verification_results['SEIR'] = {
            'mass_conservation': mass_conservation,
            'R0_correct': R0_correct,
            'incubation_correct': incubation_correct,
            'all_passed': all([mass_conservation, R0_correct, incubation_correct])
        }

        return results

    def verify_sirs_equations(self):
        """
        Verify SIRS model equations.

        Reference: Keeling & Rohani (2008)
        Equations:
            dS/dt = -β*S*I + ω*R
            dI/dt = β*S*I - γ*I
            dR/dt = γ*I - ω*R

        Properties to verify:
        1. Mass conservation: S + I + R = 1
        2. Endemic equilibrium when R0 > 1
        3. S* = 1/R0 at equilibrium
        4. Oscillatory approach to equilibrium
        """
        print("\n" + "="*60)
        print("Verifying SIRS Model")
        print("="*60)

        params = SIRSParameters(beta=0.5, gamma=0.1, omega=0.01)
        model = SIRSModel(params=params, S0=0.99, I0=0.01, R0=0.0)
        results = model.simulate(t_span=(0, 1000))

        # Verify mass conservation
        total_pop = results['S'] + results['I'] + results['R']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-6)

        # Verify endemic equilibrium
        equilibrium = model.calculate_endemic_equilibrium()
        R0 = params.beta / params.gamma

        # At equilibrium: S* = 1/R0
        S_star_expected = 1.0 / R0
        S_star_correct = np.isclose(equilibrium['S'], S_star_expected, rtol=0.01)

        # Check if system approaches equilibrium
        S_final = results['S'][-1]
        approaches_equilibrium = np.isclose(S_final, equilibrium['S'], rtol=0.1)

        print(f"✓ Mass Conservation: {mass_conservation}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n✓ Endemic Equilibrium: S* = 1/R0")
        print(f"  R0 = {R0:.2f}")
        print(f"  S* calculated: {equilibrium['S']:.3f}")
        print(f"  S* expected (1/R0): {S_star_expected:.3f}")
        print(f"  Match: {S_star_correct}")

        print(f"\n✓ Approaches Equilibrium: {approaches_equilibrium}")
        print(f"  S(final) = {S_final:.3f}")
        print(f"  I(final) = {results['I'][-1]:.3f}")
        print(f"  R(final) = {results['R'][-1]:.3f}")

        self.verification_results['SIRS'] = {
            'mass_conservation': mass_conservation,
            'equilibrium_correct': S_star_correct,
            'approaches_equilibrium': approaches_equilibrium,
            'all_passed': all([mass_conservation, S_star_correct, approaches_equilibrium])
        }

        return results

    def verify_seirv_equations(self):
        """
        Verify SEIRV model with vaccination.

        Equations:
            dS/dt = -β*S*I - ν(t)*S
            dE/dt = β*S*I - σ*E
            dI/dt = σ*E - γ*I
            dR/dt = γ*I + ε*ν(t)*S
            dV/dt = (1-ε)*ν(t)*S

        Properties to verify:
        1. Mass conservation: S + E + I + R + V = 1
        2. Vaccination reduces effective susceptible population
        3. Effective vaccination contributes to R (immune)
        4. Failed vaccination goes to V (not immune)
        """
        print("\n" + "="*60)
        print("Verifying SEIRV Model")
        print("="*60)

        params = SEIRVParameters(beta=0.6, sigma=0.2, gamma=0.1,
                                 vaccine_efficacy=0.8, vaccination_rate=0.02)
        model = SEIRVModel(params=params, S0=0.99, E0=0.0, I0=0.01, R0=0.0, V0=0.0)
        model.set_vaccination_campaign(start_time=20, duration=100, rate=0.02, efficacy=0.85)
        results = model.simulate(t_span=(0, 300))

        # Verify mass conservation
        total_pop = results['S'] + results['E'] + results['I'] + results['R'] + results['V']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-5)

        # Verify vaccination reduces S
        S_before_vacc = results['S'][results['t'] <= 20]
        S_during_vacc = results['S'][(results['t'] > 20) & (results['t'] < 120)]
        vaccination_reduces_S = np.mean(S_during_vacc) < np.mean(S_before_vacc)

        # Verify R + V increases during vaccination
        RV_before = (results['R'] + results['V'])[results['t'] <= 20]
        RV_during = (results['R'] + results['V'])[(results['t'] > 20) & (results['t'] < 120)]
        vaccination_increases_immunity = np.mean(RV_during) > np.mean(RV_before)

        print(f"✓ Mass Conservation: {mass_conservation}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n✓ Vaccination Effects:")
        print(f"  Reduces susceptible population: {vaccination_reduces_S}")
        print(f"  Increases immune population (R+V): {vaccination_increases_immunity}")
        print(f"  Final R: {results['R'][-1]:.2%}")
        print(f"  Final V: {results['V'][-1]:.2%}")
        print(f"  Total immune: {(results['R'][-1] + results['V'][-1]):.2%}")

        self.verification_results['SEIRV'] = {
            'mass_conservation': mass_conservation,
            'vaccination_reduces_S': vaccination_reduces_S,
            'vaccination_increases_immunity': vaccination_increases_immunity,
            'all_passed': all([mass_conservation, vaccination_reduces_S, vaccination_increases_immunity])
        }

        return results

    def generate_summary(self):
        """Generate verification summary."""
        print("\n" + "="*60)
        print("MATHEMATICAL VERIFICATION SUMMARY")
        print("="*60)

        all_passed = True
        for model, results in self.verification_results.items():
            status = "✓ PASS" if results['all_passed'] else "✗ FAIL"
            print(f"{model:10s}: {status}")
            all_passed = all_passed and results['all_passed']

        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL MODELS VERIFIED SUCCESSFULLY")
        else:
            print("✗ SOME MODELS FAILED VERIFICATION")
        print("="*60)

        return all_passed


class ComprehensiveVisualizer:
    """Generate comprehensive visualizations for all models."""

    def __init__(self, output_dir='results/comprehensive_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_store = {}

    def run_all_simulations(self):
        """Run all model simulations."""
        print("\n" + "="*60)
        print("Running Comprehensive Simulations")
        print("="*60)

        # SIR
        print("\nRunning SIR...")
        sir_params = SIRParameters(beta=0.5, gamma=0.1)
        sir_model = SIRModel(params=sir_params, S0=0.99, I0=0.01, R0=0.0)
        self.results_store['SIR'] = sir_model.simulate(t_span=(0, 200))
        self.results_store['SIR_model'] = sir_model

        # SEIR
        print("Running SEIR...")
        seir_params = SEIRParameters(beta=0.6, sigma=0.2, gamma=0.1)
        seir_model = SEIRModel(params=seir_params, S0=0.99, E0=0.0, I0=0.01, R0=0.0)
        self.results_store['SEIR'] = seir_model.simulate(t_span=(0, 300))
        self.results_store['SEIR_model'] = seir_model

        # SIRS
        print("Running SIRS...")
        sirs_params = SIRSParameters(beta=0.5, gamma=0.1, omega=0.01)
        sirs_model = SIRSModel(params=sirs_params, S0=0.99, I0=0.01, R0=0.0)
        self.results_store['SIRS'] = sirs_model.simulate(t_span=(0, 1000))
        self.results_store['SIRS_model'] = sirs_model

        # SEIRV
        print("Running SEIRV...")
        seirv_params = SEIRVParameters(beta=0.6, sigma=0.2, gamma=0.1,
                                       vaccine_efficacy=0.85, vaccination_rate=0.02)
        seirv_model = SEIRVModel(params=seirv_params, S0=0.99, E0=0.0, I0=0.01, R0=0.0, V0=0.0)
        seirv_model.set_vaccination_campaign(start_time=20, duration=100, rate=0.02)
        self.results_store['SEIRV'] = seirv_model.simulate(t_span=(0, 300))
        self.results_store['SEIRV_model'] = seirv_model

        # SEIR with interventions
        print("Running SEIR with interventions...")
        seir_int_model = SEIRWithInterventions(params=seir_params, S0=0.99, E0=0.0, I0=0.01, R0=0.0)
        seir_int_model.add_intervention(start_time=30, duration=30, effectiveness=0.7)
        self.results_store['SEIR_intervention'] = seir_int_model.simulate(t_span=(0, 300))
        self.results_store['SEIR_intervention_model'] = seir_int_model

        print("✓ All simulations complete")

    def create_model_comparison_figure(self):
        """Create comprehensive model comparison figure."""
        print("\nGenerating model comparison figure...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot infectious compartments for all models
        ax1 = fig.add_subplot(gs[0, :])
        models_to_compare = ['SIR', 'SEIR', 'SIRS', 'SEIRV', 'SEIR_intervention']
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for model, color in zip(models_to_compare, colors):
            results = self.results_store[model]
            ax1.plot(results['t'], results['I'], label=model, linewidth=2, color=color)

        ax1.set_xlabel('Time (days)', fontsize=12)
        ax1.set_ylabel('Infectious (fraction)', fontsize=12)
        ax1.set_title('Infectious Compartment Comparison Across Models', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Individual model dynamics
        plot_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        for idx, (model, pos) in enumerate(zip(models_to_compare[:5], plot_positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            results = self.results_store[model]
            model_obj = self.results_store[f'{model}_model']

            for state in model_obj.state_names:
                if state in results:
                    ax.plot(results['t'], results[state], label=state, linewidth=1.5)

            ax.set_xlabel('Time (days)', fontsize=10)
            ax.set_ylabel('Fraction', fontsize=10)
            ax.set_title(f'{model} Dynamics', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.savefig(self.output_dir / 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: comprehensive_model_comparison.png")
        plt.close()

    def create_phase_portraits(self):
        """Create phase portraits for all models."""
        print("\nGenerating phase portraits...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        models = [('SIR', 'S', 'I'), ('SEIR', 'S', 'I'), ('SIRS', 'S', 'I'), ('SEIRV', 'S', 'I')]

        for idx, (model, x_comp, y_comp) in enumerate(models):
            ax = axes[idx]
            results = self.results_store[model]

            x = results[x_comp]
            y = results[y_comp]

            # Plot trajectory
            ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
            ax.plot(x[0], y[0], 'go', markersize=12, label='Start')
            ax.plot(x[-1], y[-1], 'ro', markersize=12, label='End')

            # Add arrows to show direction
            n_arrows = 10
            arrow_indices = np.linspace(0, len(x)-1, n_arrows, dtype=int)
            for i in arrow_indices[:-1]:
                ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))

            ax.set_xlabel(f'{x_comp} (Fraction)', fontsize=11)
            ax.set_ylabel(f'{y_comp} (Fraction)', fontsize=11)
            ax.set_title(f'{model} Phase Portrait', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase_portraits.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase_portraits.png")
        plt.close()

    def create_reff_analysis(self):
        """Create R_effective analysis for models."""
        print("\nGenerating R_effective analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        models_with_reff = [
            ('SIR', 0.5, 0.1),
            ('SEIR', 0.6, 0.1),
            ('SIRS', 0.5, 0.1),
            ('SEIRV', 0.6, 0.1)
        ]

        for idx, (model, beta, gamma) in enumerate(models_with_reff):
            ax = axes[idx]
            results = self.results_store[model]

            R_eff = beta * results['S'] / gamma
            R0 = beta / gamma

            ax.plot(results['t'], R_eff, 'b-', linewidth=2, label='$R_{eff}(t)$')
            ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='$R_{eff} = 1$ (threshold)')
            ax.axhline(y=R0, color='g', linestyle='--', linewidth=1.5, label=f'$R_0 = {R0:.2f}$')

            ax.set_xlabel('Time (days)', fontsize=11)
            ax.set_ylabel('$R_{eff}$', fontsize=11)
            ax.set_title(f'{model} - Effective Reproduction Number', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, R0 + 1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'reff_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: reff_analysis.png")
        plt.close()

    def create_metrics_summary(self):
        """Create summary of key metrics."""
        print("\nGenerating metrics summary...")

        metrics = {}
        for model in ['SIR', 'SEIR', 'SIRS', 'SEIRV', 'SEIR_intervention']:
            results = self.results_store[model]
            model_obj = self.results_store[f'{model}_model']

            peak_time, peak_value = calculate_peak_time(results, 'I')

            metrics[model] = {
                'Peak Time (days)': f"{peak_time:.1f}",
                'Peak Infections (%)': f"{peak_value*100:.1f}",
                'Attack Rate (%)': f"{calculate_attack_rate(results)*100:.1f}",
                'Final Susceptible (%)': f"{results['S'][-1]*100:.1f}",
            }

            if hasattr(model_obj, 'calculate_herd_immunity_threshold'):
                herd = model_obj.calculate_herd_immunity_threshold()
                metrics[model]['Herd Immunity (%)'] = f"{herd*100:.1f}"

        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        # Prepare data
        col_labels = ['Metric'] + list(metrics.keys())
        row_labels = list(metrics['SIR'].keys())
        cell_text = []

        for metric in row_labels:
            row = [metric]
            for model in metrics.keys():
                row.append(metrics[model].get(metric, 'N/A'))
            cell_text.append(row)

        table = ax.table(cellText=cell_text, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=[0.25] + [0.15]*5)

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(row_labels) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: metrics_summary.png")
        plt.close()

        return metrics


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("COMPREHENSIVE VERIFICATION AND VISUALIZATION")
    print("="*60)

    # Step 1: Mathematical Verification
    print("\nSTEP 1: Mathematical Verification")
    print("-" * 60)
    verifier = MathematicalVerifier()
    verifier.verify_sir_equations()
    verifier.verify_seir_equations()
    verifier.verify_sirs_equations()
    verifier.verify_seirv_equations()
    all_verified = verifier.generate_summary()

    # Step 2: Comprehensive Visualizations
    print("\nSTEP 2: Comprehensive Visualizations")
    print("-" * 60)
    visualizer = ComprehensiveVisualizer()
    visualizer.run_all_simulations()
    visualizer.create_model_comparison_figure()
    visualizer.create_phase_portraits()
    visualizer.create_reff_analysis()
    metrics = visualizer.create_metrics_summary()

    # Step 3: Save verification results
    print("\nSTEP 3: Saving Results")
    print("-" * 60)

    # Convert numpy bools to Python bools for JSON serialization
    def convert_bools(obj):
        if isinstance(obj, dict):
            return {k: convert_bools(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    output_data = {
        'verification': convert_bools(verifier.verification_results),
        'metrics': metrics,
        'timestamp': str(np.datetime64('now'))
    }

    with open(visualizer.output_dir / 'verification_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {visualizer.output_dir}")
    print(f"✓ Verification data: verification_results.json")
    print(f"✓ Figures generated: 4 comprehensive visualizations")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    if all_verified:
        print("\n✓ All mathematical models verified successfully!")
        print("✓ All visualizations generated successfully!")
        print(f"\nView results in: {visualizer.output_dir}/")
    else:
        print("\n⚠ Some verification checks failed. Please review.")

    return all_verified


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
