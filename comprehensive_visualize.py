#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from core.base_models import SIRParameters, SEIRParameters, SIRSParameters
from models.sir_model import SIRModel
from models.seir_model import SEIRModel
from models.sirs_model import SIRSModel
from models.seirv_model import SEIRVModel, SEIRVParameters
from models.mixin_models import SEIRWithInterventions, SIRWithInterventions
from analysis.visualization import plot_comparison, plot_phase_portrait, plot_R_effective
from analysis.metrics import calculate_peak_time, calculate_attack_rate


class ComprehensiveVisualizer:

    def __init__(self):
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(f'results/comprehensive_plots_{self.timestr}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_store = {}

    def run_all_simulations(self):
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
        print("\nGenerating model comparison figure...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # infectious compartments for all models
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
        print("\nGenerating phase portraits...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        models = [('SIR', 'S', 'I'), ('SEIR', 'S', 'I'), ('SIRS', 'S', 'I'), ('SEIRV', 'S', 'I')]

        for idx, (model, x_comp, y_comp) in enumerate(models):
            ax = axes[idx]
            results = self.results_store[model]

            x = results[x_comp]
            y = results[y_comp]

            ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
            ax.plot(x[0], y[0], 'go', markersize=12, label='Start')
            ax.plot(x[-1], y[-1], 'ro', markersize=12, label='End')

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
# R_effective analysis for models
    def create_reff_analysis(self):
        
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

        # table visualization
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

        
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

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
    print("\n" + "="*60)
    print("COMPREHENSIVE VISUALIZATION")
    print("="*60)
    print("-" * 60)
    visualizer = ComprehensiveVisualizer()
    visualizer.run_all_simulations()
    visualizer.create_model_comparison_figure()
    visualizer.create_phase_portraits()
    visualizer.create_reff_analysis()
    metrics = visualizer.create_metrics_summary()

    print("\n Saving Results")
    print("-" * 60)
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
