"""
Analysis module for epidemic model results.
"""

from .metrics import (
    calculate_total_infections,
    calculate_peak_time,
    calculate_epidemic_duration,
    compare_interventions
)

from .visualization import (
    plot_comparison,
    plot_phase_portrait,
    plot_R_effective
)

# RQ1-specific visualizations (optional import)
try:
    from .rq1_visualizations import (
        create_comprehensive_timing_analysis,
        create_baseline_comparison_panel,
        create_stochastic_analysis_figure,
        create_sensitivity_heatmaps,
        create_benefit_waterfall,
        create_timing_window_analysis,
        create_vaccination_dynamics_examples
    )
    _has_rq1_viz = True
except ImportError:
    _has_rq1_viz = False

__all__ = [
    'calculate_total_infections',
    'calculate_peak_time',
    'calculate_epidemic_duration',
    'compare_interventions',
    'plot_comparison',
    'plot_phase_portrait',
    'plot_R_effective'
]

if _has_rq1_viz:
    __all__.extend([
        'create_comprehensive_timing_analysis',
        'create_baseline_comparison_panel',
        'create_stochastic_analysis_figure',
        'create_sensitivity_heatmaps',
        'create_benefit_waterfall',
        'create_timing_window_analysis',
        'create_vaccination_dynamics_examples'
    ])
