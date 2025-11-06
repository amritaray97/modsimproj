"""
Metrics and analysis functions for epidemic models.
"""

import numpy as np
from typing import Dict, List, Tuple


def calculate_total_infections(results: Dict) -> float:
    """
    Calculate total number of infections over the epidemic.

    Args:
        results: Model simulation results

    Returns:
        Total fraction of population infected
    """
    if 'R' in results:
        return results['R'][-1]
    elif 'cumulative_infections' in results:
        return results['cumulative_infections'][-1]
    else:
        raise ValueError("Cannot calculate total infections from results")


def calculate_peak_time(results: Dict, compartment: str = 'I') -> Tuple[float, float]:
    """
    Find the peak of a compartment and when it occurs.

    Args:
        results: Model simulation results
        compartment: Which compartment to analyze (default 'I')

    Returns:
        Tuple of (time of peak, peak value)
    """
    if compartment not in results:
        raise ValueError(f"Compartment {compartment} not found in results")

    peak_idx = np.argmax(results[compartment])
    return results['t'][peak_idx], results[compartment][peak_idx]


def calculate_epidemic_duration(results: Dict, threshold: float = 0.001) -> float:
    """
    Calculate how long the epidemic lasts (when I drops below threshold).

    Args:
        results: Model simulation results
        threshold: Threshold value for considering epidemic over

    Returns:
        Duration of epidemic
    """
    if 'I' not in results:
        raise ValueError("No infectious compartment in results")

    I = results['I']
    t = results['t']

    # Find when infection drops below threshold and stays there
    below_threshold = I < threshold
    if not np.any(below_threshold):
        return t[-1]  # Still above threshold at end

    first_below = np.argmax(below_threshold)
    return t[first_below]


def compare_interventions(results_list: List[Dict], labels: List[str]) -> Dict:
    """
    Compare metrics across multiple intervention scenarios.

    Args:
        results_list: List of simulation results
        labels: Labels for each scenario

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'labels': labels,
        'peak_infections': [],
        'peak_times': [],
        'total_infections': [],
        'durations': []
    }

    for results in results_list:
        peak_time, peak_value = calculate_peak_time(results, 'I')
        comparison['peak_infections'].append(peak_value)
        comparison['peak_times'].append(peak_time)
        comparison['total_infections'].append(calculate_total_infections(results))
        comparison['durations'].append(calculate_epidemic_duration(results))

    return comparison


def calculate_attack_rate(results: Dict) -> float:
    """
    Calculate the attack rate (final proportion of population infected).

    Args:
        results: Model simulation results

    Returns:
        Attack rate as a fraction
    """
    return calculate_total_infections(results)


def calculate_basic_reproduction_number(beta: float, gamma: float) -> float:
    """
    Calculate R0 from transmission and recovery rates.

    Args:
        beta: Transmission rate
        gamma: Recovery rate

    Returns:
        Basic reproduction number R0
    """
    return beta / gamma
