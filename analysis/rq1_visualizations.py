"""
RQ1-Specific Visualizations Module

Comprehensive publication-quality visualizations for vaccination timing research.
All figures include proper labels, legends, titles, and annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd


# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def create_comprehensive_timing_analysis(baseline_results, timing_results,
                                         R0_VALUES, save_path=None):
    """
    Create comprehensive multi-panel figure showing all timing analysis.

    Figure includes:
    - Attack rate vs timing (3 panels, one per R0)
    - Peak infections vs timing (3 panels)
    - Epidemic duration vs timing (3 panels)
    - Summary statistics table
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.05)

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Professional color scheme

    for idx, R0 in enumerate(R0_VALUES):
        t_starts = timing_results[R0]['t_starts']
        det_results = timing_results[R0]['deterministic']
        baseline_ar = timing_results[R0]['baseline_attack_rate']
        t_peak = baseline_results[R0]['t_peak']
        I_peak_baseline = baseline_results[R0]['I_peak']

        attack_rates = [r['attack_rate'] for r in det_results]
        peak_infections = [r['I_peak'] for r in det_results]
        t_extinctions = [r['t_extinction'] for r in det_results]

        # Find optimal
        optimal_idx = np.argmin(attack_rates)
        optimal_t = t_starts[optimal_idx]
        optimal_ar = attack_rates[optimal_idx]

        # Row 1: Attack Rate
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.plot(t_starts, attack_rates, 'o-', color=colors[idx],
                linewidth=2.5, markersize=7, markerfacecolor='white',
                markeredgewidth=2, label='With vaccination')
        ax1.axhline(baseline_ar, color='#D62828', linestyle='--',
                   linewidth=2.5, label='No vaccination', alpha=0.8)
        ax1.axvline(t_peak, color='gray', linestyle=':', linewidth=2,
                   alpha=0.6, label=f'Epidemic peak')
        ax1.plot(optimal_t, optimal_ar, '*', color='gold', markersize=22,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'Optimal (t={optimal_t:.0f}d)', zorder=10)

        # Shade optimal window (within 5% of optimal)
        threshold_ar = optimal_ar * 1.05
        effective_mask = np.array(attack_rates) <= threshold_ar
        if np.any(effective_mask):
            effective_times = t_starts[effective_mask]
            ax1.axvspan(effective_times[0], effective_times[-1],
                       alpha=0.15, color=colors[idx],
                       label='Optimal window (±5%)')

        ax1.set_xlabel('Vaccination Start Time (days)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Final Attack Rate', fontsize=11, fontweight='bold')
        ax1.set_title(f'R₀ = {R0}', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='best', framealpha=0.95, edgecolor='black')
        ax1.grid(True, alpha=0.25, linestyle='--')
        ax1.set_ylim([0, min(1.0, max(attack_rates) * 1.1)])

        # Add text box with metrics
        reduction = 100 * (baseline_ar - optimal_ar) / baseline_ar
        textstr = f'Max reduction: {reduction:.1f}%\nOptimal time: {optimal_t:.0f} days'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        # Row 2: Peak Infections
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.plot(t_starts, peak_infections, 's-', color=colors[idx],
                linewidth=2.5, markersize=6, markerfacecolor='white',
                markeredgewidth=2, label='With vaccination')
        ax2.axhline(I_peak_baseline, color='#D62828', linestyle='--',
                   linewidth=2.5, label='No vaccination', alpha=0.8)
        ax2.axvline(t_peak, color='gray', linestyle=':', linewidth=2, alpha=0.6)

        # Mark minimum peak
        min_peak_idx = np.argmin(peak_infections)
        ax2.plot(t_starts[min_peak_idx], peak_infections[min_peak_idx],
                '*', color='gold', markersize=22,
                markeredgecolor='black', markeredgewidth=1.5, zorder=10)

        ax2.set_xlabel('Vaccination Start Time (days)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Peak Infectious Fraction', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.95, edgecolor='black')
        ax2.grid(True, alpha=0.25, linestyle='--')

        # Add peak reduction text
        peak_reduction = 100 * (I_peak_baseline - min(peak_infections)) / I_peak_baseline
        textstr = f'Peak reduction: {peak_reduction:.1f}%'
        ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        # Row 3: Epidemic Duration
        ax3 = fig.add_subplot(gs[2, idx])
        ax3.plot(t_starts, t_extinctions, '^-', color=colors[idx],
                linewidth=2.5, markersize=6, markerfacecolor='white',
                markeredgewidth=2, label='With vaccination')
        ax3.axvline(t_peak, color='gray', linestyle=':', linewidth=2, alpha=0.6)

        ax3.set_xlabel('Vaccination Start Time (days)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Time to Epidemic Extinction (days)', fontsize=11, fontweight='bold')
        ax3.legend(loc='best', framealpha=0.95, edgecolor='black')
        ax3.grid(True, alpha=0.25, linestyle='--')

    fig.suptitle('Comprehensive Vaccination Timing Analysis Across R₀ Regimes',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig


def create_baseline_comparison_panel(baseline_results, R0_VALUES, save_path=None):
    """
    Create multi-panel comparison of baseline epidemics.
    Shows SEIR dynamics and key metrics for each R0.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors_compartments = {
        'S': '#0077BB', 'E': '#EE7733', 'I': '#CC3311', 'R': '#009988'
    }

    for idx, R0 in enumerate(R0_VALUES):
        res = baseline_results[R0]['results']
        t_peak = baseline_results[R0]['t_peak']
        I_peak = baseline_results[R0]['I_peak']
        attack_rate = baseline_results[R0]['attack_rate']

        # Row 1: Time series
        ax1 = fig.add_subplot(gs[0, idx])
        for comp in ['S', 'E', 'I', 'R']:
            ax1.plot(res['t'], res[comp], linewidth=2.5,
                    color=colors_compartments[comp], label=comp)
        ax1.axvline(t_peak, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.5, label='Peak')
        ax1.set_xlabel('Time (days)', fontweight='bold')
        ax1.set_ylabel('Fraction of Population', fontweight='bold')
        ax1.set_title(f'R₀ = {R0} Dynamics', fontsize=12, fontweight='bold')
        ax1.legend(loc='right', framealpha=0.95)
        ax1.grid(True, alpha=0.25)
        ax1.set_xlim([0, min(300, res['t'][-1])])

        # Add metrics annotation
        textstr = (f'Peak: {I_peak:.2%} at {t_peak:.0f}d\n'
                  f'Attack rate: {attack_rate:.2%}\n'
                  f'R₀: {R0:.1f}')
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        # Row 2: Phase portrait (S vs I)
        ax2 = fig.add_subplot(gs[1, idx])
        S = res['S']
        I = res['I']

        # Color by time
        scatter = ax2.scatter(S, I, c=res['t'], cmap='viridis',
                             s=10, alpha=0.6, edgecolors='none')
        ax2.plot(S[0], I[0], 'go', markersize=12,
                label='Start', zorder=10, markeredgecolor='black')
        ax2.plot(S[-1], I[-1], 'rs', markersize=12,
                label='End', zorder=10, markeredgecolor='black')

        ax2.set_xlabel('Susceptible (S)', fontweight='bold')
        ax2.set_ylabel('Infectious (I)', fontweight='bold')
        ax2.set_title(f'Phase Portrait (R₀={R0})', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.25)

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (days)', rotation=270, labelpad=15)

        # Row 3: R_effective over time
        ax3 = fig.add_subplot(gs[2, idx])
        R_eff = R0 * res['S']
        ax3.plot(res['t'], R_eff, linewidth=2.5, color='#0077BB',
                label='$R_{eff}(t)$')
        ax3.axhline(1, color='red', linestyle='--', linewidth=2,
                   label='$R_{eff}=1$ threshold', alpha=0.7)
        ax3.axhline(R0, color='green', linestyle=':', linewidth=2,
                   label=f'$R_0={R0}$', alpha=0.7)
        ax3.axvline(t_peak, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.5)

        ax3.set_xlabel('Time (days)', fontweight='bold')
        ax3.set_ylabel('$R_{eff}(t)$', fontweight='bold')
        ax3.set_title(f'Effective Reproduction Number (R₀={R0})', fontweight='bold')
        ax3.legend(loc='best', framealpha=0.95)
        ax3.grid(True, alpha=0.25)
        ax3.set_xlim([0, min(300, res['t'][-1])])

        # Mark when R_eff crosses 1
        cross_idx = np.argmax(R_eff < 1)
        if cross_idx > 0:
            cross_time = res['t'][cross_idx]
            ax3.plot(cross_time, 1, 'ko', markersize=10, zorder=10)
            ax3.annotate(f't={cross_time:.0f}d',
                        xy=(cross_time, 1), xytext=(cross_time+20, 1.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        fontsize=9)

    fig.suptitle('Baseline Epidemic Dynamics Without Vaccination',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig


def create_stochastic_analysis_figure(timing_results, R0_VALUES, save_path=None):
    """
    Create comprehensive stochastic variability analysis with box plots and distributions.
    """
    n_R0 = len(R0_VALUES)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, n_R0, figure=fig, hspace=0.35, wspace=0.3)

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for idx, R0 in enumerate(R0_VALUES):
        stoch_data = timing_results[R0]['stochastic']

        # Collect data
        t_starts_list = []
        attack_rates_all = []

        for condition in stoch_data:
            t_start = condition['t_start']
            replicates = condition['replicates']
            attack_rates = [r['attack_rate'] for r in replicates]

            for ar in attack_rates:
                t_starts_list.append(t_start)
                attack_rates_all.append(ar)

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame({
            't_start': t_starts_list,
            'attack_rate': attack_rates_all
        })

        # Row 1: Box plot by timing
        ax1 = fig.add_subplot(gs[0, idx])
        unique_starts = sorted(df['t_start'].unique())
        box_data = [df[df['t_start'] == t]['attack_rate'].values
                   for t in unique_starts]

        bp = ax1.boxplot(box_data, positions=unique_starts, widths=5,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(facecolor=colors[idx], alpha=0.6),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax1.set_xlabel('Vaccination Start Time (days)', fontweight='bold')
        ax1.set_ylabel('Attack Rate', fontweight='bold')
        ax1.set_title(f'R₀ = {R0}: Stochastic Variability', fontweight='bold')
        ax1.grid(True, alpha=0.25, axis='y')

        # Row 2: Mean with confidence intervals
        ax2 = fig.add_subplot(gs[1, idx])
        means = [df[df['t_start'] == t]['attack_rate'].mean()
                for t in unique_starts]
        stds = [df[df['t_start'] == t]['attack_rate'].std()
               for t in unique_starts]

        ax2.errorbar(unique_starts, means, yerr=stds, fmt='o-',
                    color=colors[idx], linewidth=2.5, markersize=8,
                    capsize=5, capthick=2, label='Mean ± SD')
        ax2.fill_between(unique_starts,
                        np.array(means) - 2*np.array(stds),
                        np.array(means) + 2*np.array(stds),
                        alpha=0.2, color=colors[idx], label='95% CI')

        ax2.set_xlabel('Vaccination Start Time (days)', fontweight='bold')
        ax2.set_ylabel('Mean Attack Rate', fontweight='bold')
        ax2.set_title(f'R₀ = {R0}: Mean with Confidence Intervals', fontweight='bold')
        ax2.legend(loc='best', framealpha=0.95)
        ax2.grid(True, alpha=0.25)

        # Row 3: Coefficient of variation
        ax3 = fig.add_subplot(gs[2, idx])
        cvs = [df[df['t_start'] == t]['attack_rate'].std() /
               df[df['t_start'] == t]['attack_rate'].mean() * 100
               for t in unique_starts]

        ax3.plot(unique_starts, cvs, 'D-', color=colors[idx],
                linewidth=2.5, markersize=7)
        ax3.set_xlabel('Vaccination Start Time (days)', fontweight='bold')
        ax3.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
        ax3.set_title(f'R₀ = {R0}: Relative Uncertainty', fontweight='bold')
        ax3.grid(True, alpha=0.25)

        # Add interpretation text
        mean_cv = np.mean(cvs)
        textstr = f'Mean CV: {mean_cv:.1f}%'
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax3.text(0.98, 0.97, textstr, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('Stochastic Variability Analysis of Vaccination Timing',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig


def create_sensitivity_heatmaps(sensitivity_results, R0_VALUES, save_path=None):
    """
    Create heatmaps showing optimal attack rate for each parameter combination.
    """
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(1, len(R0_VALUES), figsize=(15, 5))
    if len(R0_VALUES) == 1:
        axes = [axes]

    for idx, R0 in enumerate(R0_VALUES):
        sens_data = sensitivity_results[R0]['combinations']
        baseline_ar = sensitivity_results[R0]['baseline_attack_rate']

        # Build matrix: efficacy x vaccination_rate
        efficacies = sorted(set(c['efficacy'] for c in sens_data))
        vax_rates = sorted(set(c['vaccination_rate'] for c in sens_data))

        # Find optimal attack rate for each combination
        matrix = np.zeros((len(efficacies), len(vax_rates)))

        for i, eff in enumerate(efficacies):
            for j, rate in enumerate(vax_rates):
                # Find matching combination
                for combo in sens_data:
                    if combo['efficacy'] == eff and combo['vaccination_rate'] == rate:
                        attack_rates = [r['attack_rate'] for r in combo['results']]
                        matrix[i, j] = min(attack_rates)
                        break

        # Create heatmap
        ax = axes[idx]

        # Custom colormap (green = good, red = bad)
        colors_map = ['#2D7F2D', '#90EE90', '#FFFFE0', '#FFB6C1', '#DC143C']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors_map, N=n_bins)

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', origin='lower',
                      vmin=0, vmax=baseline_ar)

        # Set ticks
        ax.set_xticks(range(len(vax_rates)))
        ax.set_xticklabels([f'{r:.1%}' for r in vax_rates])
        ax.set_yticks(range(len(efficacies)))
        ax.set_yticklabels([f'{e:.0%}' for e in efficacies])

        ax.set_xlabel('Vaccination Rate (per day)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Vaccine Efficacy', fontweight='bold', fontsize=11)
        ax.set_title(f'R₀ = {R0}\nOptimal Attack Rate', fontweight='bold', fontsize=12)

        # Add text annotations
        for i in range(len(efficacies)):
            for j in range(len(vax_rates)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9,
                             fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='white',
                                     alpha=0.7, edgecolor='none'))

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attack Rate', rotation=270, labelpad=20, fontweight='bold')

        # Add baseline reference line on colorbar
        cbar.ax.axhline(baseline_ar, color='red', linestyle='--', linewidth=2)

    fig.suptitle('Parameter Sensitivity: Optimal Attack Rates',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig


def create_benefit_waterfall(baseline_results, timing_results, R0_VALUES, save_path=None):
    """
    Create waterfall chart showing vaccination benefit across R0 values.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    baseline_ars = []
    optimal_ars = []
    reductions = []

    for R0 in R0_VALUES:
        baseline_ar = baseline_results[R0]['attack_rate']
        det_results = timing_results[R0]['deterministic']
        attack_rates = [r['attack_rate'] for r in det_results]
        optimal_ar = min(attack_rates)
        reduction = baseline_ar - optimal_ar

        baseline_ars.append(baseline_ar)
        optimal_ars.append(optimal_ar)
        reductions.append(reduction)

    x = np.arange(len(R0_VALUES))
    width = 0.35

    # Baseline bars
    bars1 = ax.bar(x - width/2, baseline_ars, width, label='No Vaccination',
                   color='#D62828', alpha=0.8, edgecolor='black', linewidth=1.5)

    # With vaccination bars
    bars2 = ax.bar(x + width/2, optimal_ars, width, label='Optimal Vaccination',
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
               f'{baseline_ars[i]:.1%}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
               f'{optimal_ars[i]:.1%}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Draw reduction arrow
        arrow_x = x[i]
        ax.annotate('', xy=(arrow_x, optimal_ars[i]),
                   xytext=(arrow_x, baseline_ars[i]),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=3))

        # Add reduction percentage
        reduction_pct = 100 * reductions[i] / baseline_ars[i]
        mid_y = (baseline_ars[i] + optimal_ars[i]) / 2
        ax.text(arrow_x + 0.25, mid_y, f'-{reduction_pct:.0f}%',
               fontsize=11, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='lightyellow',
                        alpha=0.9, edgecolor='green', linewidth=2))

    ax.set_xlabel('Basic Reproduction Number (R₀)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Attack Rate', fontsize=13, fontweight='bold')
    ax.set_title('Vaccination Benefit Across R₀ Regimes',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'R₀ = {r}' for r in R0_VALUES], fontsize=11)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_ylim([0, max(baseline_ars) * 1.15])

    # Add text summary
    textstr = 'Key Finding:\n'
    if reductions[0] > reductions[-1]:
        textstr += 'Lower R₀ allows greater\nabsolute reduction'
    else:
        textstr += 'Higher R₀ shows greater\nabsolute reduction'

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9,
                edgecolor='black', linewidth=2)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right',
           bbox=props, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig


def create_timing_window_analysis(baseline_results, timing_results,
                                  R0_VALUES, save_path=None):
    """
    Create figure showing optimal timing windows and their characteristics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # Collect data
    optimal_times = []
    window_widths = []
    baseline_peaks = []

    for R0 in R0_VALUES:
        t_peak = baseline_results[R0]['t_peak']
        t_starts = timing_results[R0]['t_starts']
        det_results = timing_results[R0]['deterministic']
        attack_rates = np.array([r['attack_rate'] for r in det_results])

        optimal_idx = np.argmin(attack_rates)
        optimal_t = t_starts[optimal_idx]
        optimal_ar = attack_rates[optimal_idx]

        # Find window (within 5% of optimal)
        threshold = optimal_ar * 1.05
        window_mask = attack_rates <= threshold
        window_times = t_starts[window_mask]
        window_width = window_times[-1] - window_times[0] if len(window_times) > 0 else 0

        optimal_times.append(optimal_t)
        window_widths.append(window_width)
        baseline_peaks.append(t_peak)

    # Plot 1: Optimal timing vs R0
    ax = axes[0]
    ax.plot(R0_VALUES, optimal_times, 'o-', color='#2E86AB',
           linewidth=3, markersize=12, label='Optimal start time')
    ax.plot(R0_VALUES, baseline_peaks, 's--', color='#D62828',
           linewidth=3, markersize=10, label='Epidemic peak time')

    ax.set_xlabel('Basic Reproduction Number (R₀)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time (days)', fontweight='bold', fontsize=12)
    ax.set_title('Optimal Vaccination Timing vs. R₀', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.25)

    # Add timing relative to peak
    for i, R0 in enumerate(R0_VALUES):
        relative = optimal_times[i] - baseline_peaks[i]
        if relative < 0:
            label = f'{abs(relative):.0f}d before peak'
            color = 'green'
        else:
            label = f'{relative:.0f}d after peak'
            color = 'red'
        ax.annotate(label, xy=(R0, optimal_times[i]),
                   xytext=(R0 + 0.2, optimal_times[i] + 10),
                   fontsize=9, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor=color, linewidth=2))

    # Plot 2: Window width vs R0
    ax = axes[1]
    bars = ax.bar(R0_VALUES, window_widths, width=0.4,
                 color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    for i, (bar, width) in enumerate(zip(bars, window_widths)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{width:.0f}d',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Basic Reproduction Number (R₀)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Optimal Window Width (days)', fontweight='bold', fontsize=12)
    ax.set_title('Timing Flexibility Across R₀ Regimes', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.25, axis='y')

    # Add interpretation
    if window_widths[0] > window_widths[-1]:
        interp = 'Low R₀: More flexible timing\nHigh R₀: Narrow optimal window'
    else:
        interp = 'Timing flexibility increases with R₀'

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                edgecolor='black', linewidth=2)
    ax.text(0.98, 0.97, interp, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=props, fontweight='bold')

    # Plot 3: Relative timing (as fraction of peak time)
    ax = axes[2]
    relative_timings = np.array(optimal_times) / np.array(baseline_peaks)

    ax.plot(R0_VALUES, relative_timings, 'D-', color='#A23B72',
           linewidth=3, markersize=12)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2,
              label='At peak', alpha=0.7)
    ax.fill_between(R0_VALUES, 0.8, 1.0, alpha=0.2, color='green',
                    label='Before peak')
    ax.fill_between(R0_VALUES, 1.0, 1.2, alpha=0.2, color='red',
                    label='After peak')

    ax.set_xlabel('Basic Reproduction Number (R₀)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Optimal Time / Peak Time', fontweight='bold', fontsize=12)
    ax.set_title('Optimal Timing Relative to Epidemic Peak', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, framealpha=0.95, edgecolor='black', loc='best')
    ax.grid(True, alpha=0.25)

    # Plot 4: Summary table
    ax = axes[3]
    ax.axis('off')

    # Create table data
    table_data = []
    table_data.append(['R₀', 'Peak Time\n(days)', 'Optimal Start\n(days)',
                      'Relative to\nPeak', 'Window Width\n(days)'])

    for i, R0 in enumerate(R0_VALUES):
        relative = optimal_times[i] - baseline_peaks[i]
        rel_str = f'{abs(relative):.0f}d {"before" if relative < 0 else "after"}'
        table_data.append([
            f'{R0:.1f}',
            f'{baseline_peaks[i]:.0f}',
            f'{optimal_times[i]:.0f}',
            rel_str,
            f'{window_widths[i]:.0f}'
        ])

    table = ax.table(cellText=table_data, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#D9E2F3')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_text_props(weight='bold')

    ax.set_title('Summary of Optimal Timing Windows',
                fontweight='bold', fontsize=13, pad=20)

    fig.suptitle('Comprehensive Analysis of Optimal Vaccination Timing Windows',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig


def create_vaccination_dynamics_examples(baseline_results, timing_results,
                                         R0_VALUES, save_path=None):
    """
    Show example vaccination dynamics for early vs optimal vs late timing.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(len(R0_VALUES), 3, figure=fig, hspace=0.3, wspace=0.3)

    for r0_idx, R0 in enumerate(R0_VALUES):
        t_starts = timing_results[R0]['t_starts']
        det_results = timing_results[R0]['deterministic']
        attack_rates = [r['attack_rate'] for r in det_results]

        # Find early, optimal, and late examples
        optimal_idx = np.argmin(attack_rates)
        early_idx = 0
        late_idx = len(t_starts) - 1

        for timing_idx, label, color in [(early_idx, 'Early', '#2E86AB'),
                                          (optimal_idx, 'Optimal', '#2D7F2D'),
                                          (late_idx, 'Late', '#D62828')]:
            ax = fig.add_subplot(gs[r0_idx, [early_idx, optimal_idx, late_idx].index(timing_idx)])

            results = det_results[timing_idx]['results']
            t_start = t_starts[timing_idx]
            ar = attack_rates[timing_idx]

            # Plot SEIRV dynamics
            ax.plot(results['t'], results['S'], linewidth=2.5, label='S', color='#0077BB')
            ax.plot(results['t'], results['E'], linewidth=2.5, label='E', color='#EE7733')
            ax.plot(results['t'], results['I'], linewidth=2.5, label='I', color='#CC3311')
            ax.plot(results['t'], results['R'], linewidth=2.5, label='R', color='#009988')
            ax.plot(results['t'], results['V'], linewidth=2.5, label='V', color='#9933CC')

            # Mark vaccination start
            ax.axvline(t_start, color='black', linestyle='--', linewidth=2,
                      label=f'Vax starts (t={t_start:.0f})')

            # Shade vaccination period
            ax.axvspan(t_start, results['t'][-1], alpha=0.1, color='purple')

            ax.set_xlabel('Time (days)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Fraction of Population', fontweight='bold', fontsize=11)
            ax.set_title(f'R₀={R0}: {label} Timing (t={t_start:.0f}d)\nAttack Rate: {ar:.2%}',
                        fontweight='bold', fontsize=12, color=color)
            ax.legend(loc='right', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.25)
            ax.set_xlim([0, min(300, results['t'][-1])])

    fig.suptitle('Vaccination Dynamics: Early vs. Optimal vs. Late Timing',
                fontsize=16, fontweight='bold', y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")

    return fig
