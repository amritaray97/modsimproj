"""
Generate All Report Figures for RQ1

"""

import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')


import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import time

from analysis.rq1_visualizations import (
    create_comprehensive_timing_analysis,
    create_baseline_comparison_panel,
    create_stochastic_analysis_figure,
    create_sensitivity_heatmaps,
    create_benefit_waterfall,
    create_timing_window_analysis,
    create_vaccination_dynamics_examples
)


def main():
    print("\n" + "="*80)
    print(" "*20 + "RQ1 REPORT FIGURE GENERATION")
    print("="*80)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_dir = Path('/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj/results/rq1_vaccination_timing/')
    output_dir = Path(f'{results_dir}/report_figures_{timestr}')
    output_dir.mkdir(parents=True, exist_ok=True)


    # results_dir.append{'/report_figures_{timestr}'}
    # output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # if results exist
    if not results_dir.exists():
        print("\n ERROR: RQ1 results not found!")
        print("Please run one of these first:")
        print("  - python experiments/rq1_vaccination_timing.py")
        print("  - python experiments/rq1_vaccination_timing_quick.py")
        return

    print("\nLoading results...")
    try:
        with open(results_dir / 'phase1_baseline.pkl', 'rb') as f:
            baseline_results = pickle.load(f)
            print(" Baseline results loaded")

        with open(results_dir / 'phase2_timing_sweep.pkl', 'rb') as f:
            timing_results = pickle.load(f)
            print(" Timing sweep results loaded")
        try:
            with open(results_dir / 'phase3_sensitivity.pkl', 'rb') as f:
                sensitivity_results = pickle.load(f)
                has_sensitivity = True
                print(" Sensitivity results loaded")
        except FileNotFoundError:
            sensitivity_results = None
            has_sensitivity = False
            print(" Sensitivity results not found (using quick version data)")

    except Exception as e:
        print(f"\n ERROR loading results: {e}")
        return

    R0_VALUES = list(baseline_results.keys())
    print(f"\nR₀ values found: {R0_VALUES}")

    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    figures_generated = []

    # Figure 1: Comprehensive Timing Analysis
    print("\n[1/7] Creating comprehensive timing analysis...")
    fig1_path = f"{output_dir}/figure1_comprehensive_timing_analysis.png"
    try:
        create_comprehensive_timing_analysis(
            baseline_results, timing_results, R0_VALUES, save_path=fig1_path
        )
        figures_generated.append('Figure 1: Comprehensive Timing Analysis')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

    # Figure 2: Baseline Comparison
    print("\n[2/7] Creating baseline comparison panel...")
    fig2_path = f'{output_dir}/figure2_baseline_comparison.png'
    try:
        create_baseline_comparison_panel(
            baseline_results, R0_VALUES, save_path=fig2_path
        )
        figures_generated.append('Figure 2: Baseline Epidemic Dynamics')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

    # Figure 3: Stochastic Analysis
    print("\n[3/7] Creating stochastic variability analysis...")
    fig3_path = f'{output_dir}/figure3_stochastic_analysis.png'
    try:
        # Check if stochastic data exists
        if len(timing_results[R0_VALUES[0]]['stochastic']) > 0:
            create_stochastic_analysis_figure(
                timing_results, R0_VALUES, save_path=fig3_path
            )
            figures_generated.append('Figure 3: Stochastic Variability')
            plt.close()
        else:
            print("  ⚠ No stochastic data available (quick version)")
    except Exception as e:
        print(f"Error: {e}")

    # Figure 4: Sensitivity Heatmaps
    if has_sensitivity and sensitivity_results is not None:
        print("\n[4/7] Creating parameter sensitivity heatmaps...")
        fig4_path = f'{output_dir}/figure4_sensitivity_heatmaps.png'
        try:
            create_sensitivity_heatmaps(
                sensitivity_results, R0_VALUES, save_path=fig4_path
            )
            figures_generated.append('Figure 4: Parameter Sensitivity Heatmaps')
            plt.close()
        except Exception as e:
            print(f" Error: {e}")
    else:
        print("\n[4/7] Skipping sensitivity heatmaps (not available)")

    # Figure 5: Benefit Waterfall
    print("\n[5/7] Creating vaccination benefit waterfall chart...")
    fig5_path = f'{output_dir}/figure5_benefit_waterfall.png'
    try:
        create_benefit_waterfall(
            baseline_results, timing_results, R0_VALUES, save_path=fig5_path
        )
        figures_generated.append('Figure 5: Vaccination Benefit Analysis')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

    # Figure 6: Timing Window Analysis
    print("\n[6/7] Creating timing window analysis...")
    fig6_path = output_dir / 'figure6_timing_windows.png'
    try:
        create_timing_window_analysis(
            baseline_results, timing_results, R0_VALUES, save_path=fig6_path
        )
        figures_generated.append('Figure 6: Optimal Timing Windows')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

    # Figure 7: Vaccination Dynamics Examples
    print("\n[7/7] Creating vaccination dynamics examples...")
    fig7_path = f'{output_dir}/figure7_vaccination_dynamics.png'
    try:
        create_vaccination_dynamics_examples(
            baseline_results, timing_results, R0_VALUES, save_path=fig7_path
        )
        figures_generated.append('Figure 7: Vaccination Dynamics Examples')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSuccessfully generated {len(figures_generated)} figures:")
    for i, fig_name in enumerate(figures_generated, 1):
        print(f"  {i}. {fig_name}")

    print(f"\n All figures saved to: {output_dir}")

    # index files
    print("\nCreating figure index...")
    index_path = f'{output_dir}/FIGURE_INDEX.txt'
    with open(index_path, 'w') as f:
        f.write("RQ1 VACCINATION TIMING RESEARCH - FIGURE INDEX\n")
        f.write("=" * 70 + "\n\n")

        if 'Figure 1: Comprehensive Timing Analysis' in figures_generated:
            f.write("Figure 1: Comprehensive Timing Analysis\n")
            f.write("  File: figure1_comprehensive_timing_analysis.png\n")
            f.write("  Description: 9-panel figure showing attack rate, peak infections,\n")
            f.write("               and epidemic duration vs vaccination timing for each R₀\n")

        if 'Figure 2: Baseline Epidemic Dynamics' in figures_generated:
            f.write("Figure 2: Baseline Epidemic Dynamics\n")
            f.write("  File: figure2_baseline_comparison.png\n")
            f.write("  Description: SEIR dynamics, phase portraits, and R_eff plots\n")
            f.write("               for baseline epidemics (no vaccination)\n")

        if 'Figure 3: Stochastic Variability' in figures_generated:
            f.write("Figure 3: Stochastic Variability Analysis\n")
            f.write("  File: figure3_stochastic_analysis.png\n")
            f.write("  Description: Box plots, confidence intervals, and coefficient\n")
            f.write("               of variation for stochastic simulations\n")

        if 'Figure 4: Parameter Sensitivity Heatmaps' in figures_generated:
            f.write("Figure 4: Parameter Sensitivity Heatmaps\n")
            f.write("  File: figure4_sensitivity_heatmaps.png\n")
            f.write("  Description: Heatmaps showing optimal attack rates for different\n")
            f.write("               vaccine efficacy and vaccination rate combinations\n")

        if 'Figure 5: Vaccination Benefit Analysis' in figures_generated:
            f.write("Figure 5: Vaccination Benefit Waterfall Chart\n")
            f.write("  File: figure5_benefit_waterfall.png\n")
            f.write("  Description: Bar chart comparing baseline vs optimal vaccination\n")
            f.write("               showing absolute and relative reductions\n")

        if 'Figure 6: Optimal Timing Windows' in figures_generated:
            f.write("Figure 6: Optimal Timing Windows Analysis\n")
            f.write("  File: figure6_timing_windows.png\n")
            f.write("  Description: 4-panel analysis of optimal timing, window widths,\n")
            f.write("               relative timing, and summary table\n")

        if 'Figure 7: Vaccination Dynamics Examples' in figures_generated:
            f.write("Figure 7: Vaccination Dynamics Examples\n")
            f.write("  File: figure7_vaccination_dynamics.png\n")
            f.write("  Description: SEIRV time series for early, optimal, and late\n")
            f.write("               vaccination scenarios\n")
    

    print(f"Figure index saved to: {index_path}")


if __name__ == "__main__":
    main()
