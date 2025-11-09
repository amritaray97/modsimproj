#!/usr/bin/env python3
"""
Main simulation runner for the Epidemic Simulator

This script reads a configuration file and runs the specified epidemic simulation
with all configured parameters, interventions, and visualizations.
Usage:
    python run_simulation.py --config configs/sir_basic.json
    python run_simulation.py --config configs/seir_with_interventions.json
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import ConfigLoader
from core.base_models import SIRParameters, SEIRParameters, SIRSParameters, SIRDParameters
from models.sir_model import SIRModel
from models.seir_model import SEIRModel
from models.sirs_model import SIRSModel
from models.sird_model import SIRDModel
from models.seirv_model import SEIRVModel, SEIRVParameters
from models.mixin_models import SEIRWithInterventions, SIRWithInterventions

"""
Main simulation runner that executes configured epidemic simulations.

"""
class SimulationRunner:
    

    def __init__(self, config_path: str):
        self.config = ConfigLoader(config_path)
        self.model = None
        self.results = None

    def create_model(self):
        model_type = self.config.get_model_type()
        params_dict = self.config.get_parameters()
        initial_conditions = self.config.get_initial_conditions()
        interventions = self.config.get_interventions()

        if model_type == 'SIR':
            params = SIRParameters(**params_dict)
            if interventions:
                self.model = SIRWithInterventions(
                    params=params,
                    S0=initial_conditions['S0'],
                    I0=initial_conditions['I0'],
                    R0=initial_conditions['R0']
                )
                self._add_interventions(self.model, interventions)
            else:
                self.model = SIRModel(
                    params=params,
                    S0=initial_conditions['S0'],
                    I0=initial_conditions['I0'],
                    R0=initial_conditions['R0']
                )

        elif model_type == 'SEIR':
            params = SEIRParameters(**params_dict)
            if interventions:
                self.model = SEIRWithInterventions(
                    params=params,
                    S0=initial_conditions['S0'],
                    E0=initial_conditions['E0'],
                    I0=initial_conditions['I0'],
                    R0=initial_conditions['R0']
                )
                self._add_interventions(self.model, interventions)
            else:
                self.model = SEIRModel(
                    params=params,
                    S0=initial_conditions['S0'],
                    E0=initial_conditions['E0'],
                    I0=initial_conditions['I0'],
                    R0=initial_conditions['R0']
                )

        elif model_type == 'SIRS':
            params = SIRSParameters(**params_dict)
            self.model = SIRSModel(
                params=params,
                S0=initial_conditions['S0'],
                I0=initial_conditions['I0'],
                R0=initial_conditions['R0']
            )

        elif model_type == 'SIRD':
            params = SIRDParameters(**params_dict)
            self.model = SIRDModel(
                params=params,
                S0=initial_conditions['S0'],
                I0=initial_conditions['I0'],
                R0=initial_conditions['R0'],
                D0=initial_conditions.get('D0', 0.0)
            )

        elif model_type == 'SEIRV':
            params = SEIRVParameters(**params_dict)
            self.model = SEIRVModel(
                params=params,
                S0=initial_conditions['S0'],
                E0=initial_conditions['E0'],
                I0=initial_conditions['I0'],
                R0=initial_conditions['R0'],
                V0=initial_conditions['V0']
            )

            vaccination_config = self.config.get_vaccination_settings()
            if vaccination_config and vaccination_config.get('campaign_type') == 'timed':
                self.model.set_vaccination_campaign(
                    start_time=vaccination_config['start_time'],
                    duration=vaccination_config['duration'],
                    rate=vaccination_config['rate'],
                    efficacy=vaccination_config.get('efficacy', params.vaccine_efficacy)
                )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Created {model_type} model")
        self._print_model_info()

    def _add_interventions(self, model, interventions):
        for intervention in interventions:
            model.add_intervention(
                start_time=intervention['start_time'],
                duration=intervention['duration'],
                effectiveness=intervention['effectiveness'],
                intervention_type=intervention.get('intervention_type', 'reduction')
            )
            print(f"  - Added intervention: {intervention.get('name', 'Unnamed')} "
                  f"(t={intervention['start_time']}-{intervention['start_time'] + intervention['duration']}, "
                  f"eff={intervention['effectiveness']:.0%})")

    def _print_model_info(self):
        print("\nModel Parameters:")
        params = self.model.params
        print(f"  β (beta): {params.beta:.3f}")
        print(f"  γ (gamma): {params.gamma:.3f}")

        if hasattr(params, 'sigma'):
            print(f"  σ (sigma): {params.sigma:.3f}")
            print(f"  Incubation period: {params.incubation_period:.1f} days")

        if hasattr(params, 'omega'):
            print(f"  ω (omega): {params.omega:.3f}")

        if hasattr(params, 'vaccine_efficacy'):
            print(f"  Vaccine efficacy: {params.vaccine_efficacy:.1%}")
            print(f"  Vaccination rate: {params.vaccination_rate:.3f}")

        print(f"  R₀: {params.R0:.2f}")

        print("\nInitial Conditions:")
        initial = self.config.get_initial_conditions()
        for state, value in initial.items():
            print(f"  {state}: {value:.3f}")

    def run_simulation(self):
        sim_settings = self.config.get_simulation_settings()

        t_span = (sim_settings['t_start'], sim_settings['t_end'])
        t_eval = np.linspace(sim_settings['t_start'], sim_settings['t_end'],
                            sim_settings['num_points'])

        print(f"\n{'='*60}")
        print(f"Running simulation from t={t_span[0]} to t={t_span[1]}")
        print(f"{'='*60}")

        # stochastic settings
        stochastic = self.config.get_stochastic_settings()
        if stochastic and stochastic.get('enabled', False):
            self._run_stochastic_simulation(t_span, t_eval, stochastic)
        else:
            self.results = self.model.simulate(
                t_span=t_span,
                t_eval=t_eval,
                method=sim_settings.get('method', 'RK45')
            )
            print("✓ Simulation complete")

    def _run_stochastic_simulation(self, t_span, t_eval, stochastic_config):
        num_realizations = stochastic_config.get('num_realizations', 5)
        noise_level = stochastic_config.get('noise_level', 0.05)
        seed = stochastic_config.get('seed', None)

        if seed is not None:
            np.random.seed(seed)

        print(f"Running {num_realizations} stochastic realizations with noise level {noise_level}")

        self.results = {'t': t_eval, 'realizations': []}

        for i in range(num_realizations):
            result = self.model.simulate(t_span=t_span, t_eval=t_eval)

            # Adding noise to all compartments
            noisy_result = {'t': result['t']}
            for state in self.model.state_names:
                noise = np.random.normal(0, noise_level, len(result[state]))
                noisy_result[state] = np.clip(result[state] + noise, 0, 1)

            self.results['realizations'].append(noisy_result)
            print(f"Completed realization {i+1}/{num_realizations}")

        # mean and std for main results
        for state in self.model.state_names:
            values = np.array([r[state] for r in self.results['realizations']])
            self.results[state] = np.mean(values, axis=0)
            self.results[f'{state}_std'] = np.std(values, axis=0)

        print("Stochastic simulation complete")

    def calculate_metrics(self):
        print(f"\n{'='*60}")
        print("Epidemic Metrics")
        print(f"{'='*60}")

        if 'realizations' in self.results:
            results = {k: v for k, v in self.results.items() if k not in ['realizations']}
        else:
            results = self.results

        # Peak infection
        if 'I' in results:
            peak_time, peak_infections = self.model.calculate_peak_infection(results)
            print(f"Peak Infections: {peak_infections:.2%} at day {peak_time:.1f}")

        # Attack rate
        try:
            attack_rate = self.model.calculate_attack_rate(results)
            print(f"Attack Rate: {attack_rate:.2%}")
        except:
            pass

        # Herd immunity threshold
        if hasattr(self.model, 'calculate_herd_immunity_threshold'):
            herd_immunity = self.model.calculate_herd_immunity_threshold()
            print(f"Herd Immunity Threshold: {herd_immunity:.2%}")

        # SIRS endemic equilibrium
        if hasattr(self.model, 'calculate_endemic_equilibrium'):
            equilibrium = self.model.calculate_endemic_equilibrium()
            print(f"\nEndemic Equilibrium:")
            for state, value in equilibrium.items():
                print(f"  {state}: {value:.3f}")

        # SEIRV vaccination metrics
        if hasattr(self.model, 'calculate_total_vaccinated'):
            total_vacc = self.model.calculate_total_vaccinated(results)
            print(f"Total Vaccinated: {total_vacc:.2%}")

    def save_results(self):
        output_settings = self.config.get_output_settings()

        if not output_settings['save_plots']:
            return

        output_dir = Path(output_settings['output_dir'])
        output_dir.mkdir(exist_ok=True)

        config_name = Path(self.config.config_path).stem
        model_type = self.config.get_model_type().lower()

        fig, ax = plt.subplots(figsize=(12, 7))

        stochastic = self.config.get_stochastic_settings()
        if stochastic and stochastic.get('enabled', False):
            self._plot_stochastic_results(ax)
        else:
            self.model.plot_dynamics(results=self.results, ax=ax)

        interventions = self.config.get_interventions()
        if interventions:
            for intervention in interventions:
                start = intervention['start_time']
                end = start + intervention['duration']
                ax.axvspan(start, end, alpha=0.2, color='gray',
                          label=f"Intervention: {intervention.get('name', 'Unnamed')}")

        # vaccination marker
        vaccination = self.config.get_vaccination_settings()
        if vaccination:
            start = vaccination['start_time']
            end = start + vaccination['duration']
            ax.axvspan(start, end, alpha=0.2, color='purple',
                      label='Vaccination Campaign')

        ax.legend()
        plt.tight_layout()

        # Save plot
        filename = f"{config_name}_{model_type}.{output_settings['plot_format']}"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=output_settings['dpi'])
        print(f"\n Plot saved to: {filepath}")

        if output_settings['show_plots']:
            plt.show()
        else:
            plt.close()

    def _plot_stochastic_results(self, ax):
        results = self.results

        for state in self.model.state_names:
            mean = results[state]
            std = results[f'{state}_std']

            color = {'S': 'blue', 'E': 'yellow', 'I': 'red', 'R': 'green', 'V': 'purple'}.get(state, 'black')
            ax.plot(results['t'], mean, label=self.model._get_label(state), color=color, linewidth=2)

            # uncertainty band
            ax.fill_between(results['t'], mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction of Population')
        ax.set_title(f'{self.model.__class__.__name__} Dynamics (Stochastic)')
        ax.grid(True, alpha=0.3)

    def run(self):
        print(f"\n{'='*60}")
        print(f"Epidemic Simulator - Configuration Run")
        print(f"{'='*60}")
        print(f"Config file: {self.config.config_path}")
        print(f"Description: {self.config.config.get('model', {}).get('description', 'N/A')}")

        self.create_model()
        self.run_simulation()
        self.calculate_metrics()
        self.save_results()

        print(f"\n{'='*60}")
        print("Simulation complete!")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run epidemic simulations from configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to JSON configuration file'
    )

    args = parser.parse_args()

    try:
        runner = SimulationRunner(args.config)
        runner.run()
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
