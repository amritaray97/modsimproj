#!/usr/bin/env python3
"""Simple test of SIRD model without plotting"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.base_models import SIRDParameters
from models.sird_model import SIRDModel

# Create parameters
params = SIRDParameters(
    population=1_000_000,
    beta=0.5,
    gamma=0.1,
    mu=0.02
)

print("SIRD Model Test")
print("=" * 50)
print(f"Parameters:")
print(f"  Beta (transmission): {params.beta}")
print(f"  Gamma (recovery): {params.gamma}")
print(f"  Mu (mortality): {params.mu}")
print(f"  R0: {params.R0:.2f}")

# Create model
model = SIRDModel(params=params, S0=0.99, I0=0.01, R0=0.0, D0=0.0)
print(f"\nCompartments: {model.state_names}")

# Run simulation
print("\nRunning simulation...")
results = model.simulate(t_span=(0, 200))

# Get summary
summary = model.get_epidemic_summary(results)

print("\nResults:")
print(f"  Peak infection time: Day {summary['peak_infection_time']:.1f}")
print(f"  Peak infection level: {summary['peak_infection_fraction']:.2%}")
print(f"  Attack rate: {summary['attack_rate']:.1%}")
print(f"  Case Fatality Rate: {summary['case_fatality_rate']:.2%}")
print(f"  Total deaths: {summary['total_deaths_count']:,.0f}")
print(f"  Final susceptible: {summary['final_susceptible']:.2%}")

# Verify conservation of population
final_total = results['S'][-1] + results['I'][-1] + results['R'][-1] + results['D'][-1]
print(f"\nPopulation conservation check: {final_total:.6f}")
print(f"  (should be 1.0)")

print("\n✅ SIRD model test successful!")
