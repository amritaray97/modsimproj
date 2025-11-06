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

__all__ = [
    'calculate_total_infections',
    'calculate_peak_time',
    'calculate_epidemic_duration',
    'compare_interventions',
    'plot_comparison',
    'plot_phase_portrait',
    'plot_R_effective'
]
