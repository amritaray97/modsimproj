"""
Visualization utilities for epidemic model results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_comparison(results_list: List[Dict],
                   labels: List[str],
                   compartment: str = 'I',
                   title: Optional[str] = None,
                   figsize: tuple = (12, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    for results, label in zip(results_list, labels):
        if compartment in results:
            ax.plot(results['t'], results[compartment], label=label, linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel(f'{compartment} (Fraction of Population)')
    ax.set_title(title or f'{compartment} Compartment Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_phase_portrait(results: Dict,
                       x_compartment: str = 'S',
                       y_compartment: str = 'I',
                       figsize: tuple = (8, 8)) -> plt.Figure:

    if x_compartment not in results or y_compartment not in results:
        raise ValueError(f"Compartments {x_compartment} and {y_compartment} must be in results")

    fig, ax = plt.subplots(figsize=figsize)

    x = results[x_compartment]
    y = results[y_compartment]

    # trajectory with color gradient showing time
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    from matplotlib.collections import LineCollection

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(results['t'])

    ax.add_collection(lc)
    ax.autoscale()

    # start and end
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')

    ax.set_xlabel(f'{x_compartment} (Fraction)')
    ax.set_ylabel(f'{y_compartment} (Fraction)')
    ax.set_title(f'Phase Portrait: {y_compartment} vs {x_compartment}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Time')

    return fig


def plot_R_effective(results: Dict,
                    beta: float,
                    gamma: float,
                    figsize: tuple = (10, 6)) -> plt.Figure:
    if 'S' not in results:
        raise ValueError("Results must contain 'S' compartment")

    fig, ax = plt.subplots(figsize=figsize)

    R_eff = beta * results['S'] / gamma
    R0 = beta / gamma

    ax.plot(results['t'], R_eff, label='$R_{eff}(t)$', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', label='$R_{eff} = 1$ (threshold)')
    ax.axhline(y=R0, color='g', linestyle='--', label=f'$R_0 = {R0:.2f}$')

    ax.set_xlabel('Time')
    ax.set_ylabel('Effective Reproduction Number')
    ax.set_title('Effective Reproduction Number Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_multi_compartment(results: Dict,
                          compartments: Optional[List[str]] = None,
                          figsize: tuple = (12, 8)) -> plt.Figure:
    
    if compartments is None:
        compartments = [k for k in results.keys() if k != 't']

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(compartments)))

    for compartment, color in zip(compartments, colors):
        if compartment in results:
            ax.plot(results['t'], results[compartment],
                   label=compartment, color=color, linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of Population')
    ax.set_title('Epidemic Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
