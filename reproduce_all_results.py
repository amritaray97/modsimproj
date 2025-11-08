#!/usr/bin/env python3


import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json
from typing import Optional


class ReproducibilityRunner:

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.start_time = time.time()
        self.results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': [],
            'files_generated': [],
            'total_runtime': None
        }

    def print_header(self, text):
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80 + "\n")

    def print_step(self, step_num, total_steps, description):
        print(f"\n[{step_num}/{total_steps}] {description}")
        print("-" * 80)

    def run_command(self, cmd, description, cwd=None):
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per step
            )

            if result.returncode == 0:
                self.results['steps_completed'].append(description)
                print(f"{description} completed successfully")
                if result.stdout:
                    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                return True
            else:
                self.results['steps_failed'].append(description)
                print(f"{description} failed!")
                print(f"Error output:\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.results['steps_failed'].append(f"{description} (timeout)")
            print(f"{description} timed out!")
            return False
        except Exception as e:
            self.results['steps_failed'].append(f"{description} (error: {e})")
            print(f"{description} encountered error: {e}")
            return False

    
    # Math checker to verify mathematical validity of the results
    # Can be commmented out later, this is more for debugging

    def step1_mathematical_verification(self):
        self.print_step(1, 6, "Mathematical Verification & Comprehensive Visualization")

        # For both math checker and visualizer:
        success = self.run_command(
            [sys.executable, 'verify_and_visualize.py'],
            "Mathematical verification"
        )
        
        # For only math checker:
        # success = self.run_command(
        #    [sys.executable, 'math_verify.py'],
        #    "Math Checker"
        #)
        
        # For only visualization:
        #success = self.run_command(
        #    [sys.executable, 'comprehensive_visualize.py'],
        #    "Comprehensive Visualization"
        #)

        if success:
            output_dir = self.project_root / 'results' / 'comprehensive_analysis'
            if output_dir.exists():
                for file in output_dir.glob('*'):
                    self.results['files_generated'].append(str(file.relative_to(self.project_root)))

        return success

    def step2_basic_simulations(self):
        self.print_step(2, 6, "Configuration-Based Simulations")

        configs = [
            'configs/sir_basic.json',
            'configs/seir_with_interventions.json',
            'configs/sirs_endemic.json',
            'configs/seirv_vaccination.json',
            'configs/seir_stochastic.json'
        ]

        all_success = True
        for config in configs:
            config_path = self.project_root / config
            if not config_path.exists():
                print(f"Config file not found: {config}")
                continue

            config_name = config_path.stem
            print(f"\n  Running simulation: {config_name}")

            success = self.run_command(
                [sys.executable, 'run_simulation.py', '--config', str(config_path)],
                f"Simulation: {config_name}"
            )

            if success:
                # Track generated plot
                results_dir = self.project_root / 'results'
                for plot_file in results_dir.glob(f"{config_name}*.png"):
                    self.results['files_generated'].append(str(plot_file.relative_to(self.project_root)))
            else:
                all_success = False

        return all_success

    def step3_model_comparison(self):
        self.print_step(3, 6, "Model Comparison Analysis")

        experiment_file = self.project_root / 'experiments' / 'model_comparison.py'
        if not experiment_file.exists():
            print(f"Experiment file not found: {experiment_file}")
            return True  # Not critical

        success = self.run_command(
            [sys.executable, str(experiment_file)],
            "Model comparison"
        )

        if success:
            results_dir = self.project_root / 'results'
            for file in results_dir.glob('model_comparison*'):
                self.results['files_generated'].append(str(file.relative_to(self.project_root)))

        return success

    def step4_vaccination_timing_analysis(self):
        self.print_step(4, 6, "RQ1: Vaccination Timing Optimization (Quick Version)")
        print("Note: Running short version.")
        print("For full analysis: python3 experiments/rq1_vaccination_timing.py")

        experiment_file = self.project_root / 'experiments' / 'rq1_vaccination_timing_quick.py'
        if not experiment_file.exists():
            print(f"Experiment file not found: {experiment_file}")
            return True  # Not critical

        success = self.run_command(
            [sys.executable, str(experiment_file)],
            "RQ1 vaccination timing (quick)"
        )

        if success:
            results_dir = self.project_root / 'results'
            for file in results_dir.rglob('rq1*'):
                if file.is_file():
                    self.results['files_generated'].append(str(file.relative_to(self.project_root)))

        return success

    def step5_generate_report_figures(self):
        self.print_step(5, 6, "Report Figure Generation")

        experiment_file = self.project_root / 'experiments' / 'generate_report_figures.py'
        if not experiment_file.exists():
            print(f"⚠ Report figures script not found: {experiment_file}")
            return True  # Not critical

        success = self.run_command(
            [sys.executable, str(experiment_file)],
            "Report figure generation"
        )

        if success:
            results_dir = self.project_root / 'results'
            for file in results_dir.rglob('report_figures/*'):
                if file.is_file():
                    self.results['files_generated'].append(str(file.relative_to(self.project_root)))

        return success

    def step6_collect_results(self):
        self.print_step(6, 6, "Collecting Results and Generating Summary")

        # files by type
        file_counts = {
            'PNG images': len([f for f in self.results['files_generated'] if f.endswith('.png')]),
            'JSON data': len([f for f in self.results['files_generated'] if f.endswith('.json')]),
            'PKL data': len([f for f in self.results['files_generated'] if f.endswith('.pkl')]),
            'CSV data': len([f for f in self.results['files_generated'] if f.endswith('.csv')])
        }

        print("\nFiles generated by type:")
        for file_type, count in file_counts.items():
            print(f"  {file_type}: {count}")

        # Summary
        self.results['file_counts'] = file_counts
        summary_file = self.project_root / 'results' / 'reproducibility_summary.json'
        summary_file.parent.mkdir(exist_ok=True, parents=True)

        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Summary saved to: {summary_file}")
        self.results['files_generated'].append(str(summary_file.relative_to(self.project_root)))

        return True

    def generate_final_report(self):
        self.print_header("REPRODUCIBILITY RUN COMPLETE")

        elapsed_time = time.time() - self.start_time
        self.results['total_runtime'] = f"{elapsed_time/60:.1f} minutes"

        print(f"Start time: {self.results['start_time']}")
        print(f"Total runtime: {self.results['total_runtime']}")
        print(f"\nSteps completed: {len(self.results['steps_completed'])}")
        print(f"Steps failed: {len(self.results['steps_failed'])}")
        print(f"Files generated: {len(self.results['files_generated'])}")

        if self.results['steps_failed']:
            print("\n⚠ Failed steps:")
            for step in self.results['steps_failed']:
                print(f"  - {step}")

        print("\n" + "="*80)
        print("\n" + "="*80)

        # Return success if no critical failures
        return len(self.results['steps_failed']) == 0

    def run(self):
        """Execute all reproducibility steps."""
        self.print_header("EPIDEMIC SIMULATOR - REPRODUCIBILITY RUN")
        print(f"Project root: {self.project_root}")
        print(f"Python: {sys.executable}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Execute all steps
        steps = [
            self.step1_mathematical_verification,
            self.step2_basic_simulations,
            self.step3_model_comparison,
            self.step4_vaccination_timing_analysis,
            self.step5_generate_report_figures,
            self.step6_collect_results
        ]

        for step in steps:
            try:
                step()
            except Exception as e:
                print(f"\n✗ Step failed with exception: {e}")
                import traceback
                traceback.print_exc()

        # Generate final report
        success = self.generate_final_report()

        return 0 if success else 1


def main():
    """Main entry point."""
    runner = ReproducibilityRunner()
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
