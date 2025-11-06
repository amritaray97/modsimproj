# Epidemic Simulator

A comprehensive Python framework for simulating epidemic dynamics using compartmental models.

## Features

- **Multiple Epidemic Models**: SIR, SEIR, SIRS with extensible architecture
- **Intervention Modeling**: Simulate lockdowns, social distancing, vaccination campaigns
- **Stochastic Simulations**: Add noise to capture uncertainty and random effects
- **Analysis Tools**: Built-in metrics and visualization utilities
- **Modular Design**: Easy to extend with new models and features

## Project Structure

```
modsimproj/
├── core/                       # Core framework components
│   ├── base_models.py         # Abstract base classes and parameters
│   └── mixins.py              # Mixins for stochastic, interventions, networks
│
├── models/                     # Concrete model implementations
│   ├── sir_model.py           # SIR model
│   ├── seir_model.py          # SEIR model
│   ├── sirs_model.py          # SIRS model
│   └── mixin_models.py        # Models with enhanced features
│
├── interventions/             # Intervention strategies
│   └── strategies.py          # Lockdown, social distancing, etc.
│
├── analysis/                  # Analysis and visualization tools
│   ├── metrics.py             # Analysis metrics
│   └── visualization.py       # Plotting utilities
│
├── experiments/               # Example simulations
│   ├── basic_sir_simulation.py
│   ├── seir_with_interventions.py
│   ├── model_comparison.py
│   └── stochastic_simulation.py
│
└── results/                   # Output directory for plots and data
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd modsimproj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic SIR Simulation

```python
from core.base_models import SIRParameters
from models.sir_model import SIRModel

# Set up parameters
params = SIRParameters(
    beta=0.5,      # Transmission rate
    gamma=0.1      # Recovery rate
)

# Create and run model
model = SIRModel(params=params, S0=0.99, I0=0.01, R0=0.0)
results = model.simulate(t_span=(0, 200))

# Plot results
model.plot_dynamics(results)
```

### SEIR with Interventions

```python
from core.base_models import SEIRParameters
from models.mixin_models import SEIRWithInterventions

# Set up parameters
params = SEIRParameters(beta=0.6, sigma=0.2, gamma=0.1)

# Create model
model = SEIRWithInterventions(params=params, S0=0.99, E0=0.0, I0=0.01, R0=0.0)

# Add intervention (lockdown from day 30-60 with 70% effectiveness)
model.add_intervention(start_time=30, duration=30, effectiveness=0.7)

# Run simulation
results = model.simulate(t_span=(0, 300))
```

## Models

### SIR Model
Classic compartmental model with Susceptible, Infectious, and Recovered states.

**Equations:**
- dS/dt = -β·S·I
- dI/dt = β·S·I - γ·I
- dR/dt = γ·I

### SEIR Model
Extends SIR with Exposed (latent) compartment for diseases with incubation period.

**Equations:**
- dS/dt = -β·S·I
- dE/dt = β·S·I - σ·E
- dI/dt = σ·E - γ·I
- dR/dt = γ·I

### SIRS Model
SIR model with waning immunity, allowing reinfection.

**Equations:**
- dS/dt = -β·S·I + ω·R
- dI/dt = β·S·I - γ·I
- dR/dt = γ·I - ω·R

## Running Examples

Run the example simulations:

```bash
# Basic SIR simulation
python experiments/basic_sir_simulation.py

# SEIR with interventions
python experiments/seir_with_interventions.py

# Compare different models
python experiments/model_comparison.py

# Stochastic simulation
python experiments/stochastic_simulation.py
```

Results will be saved in the `results/` directory.

## Key Concepts

### Parameters

- **β (beta)**: Transmission rate - probability of disease transmission per contact
- **γ (gamma)**: Recovery rate - rate at which infected individuals recover
- **σ (sigma)**: Incubation rate - rate of progression from exposed to infectious
- **ω (omega)**: Waning immunity rate - rate at which immunity is lost

### Metrics

- **R₀ (Basic Reproduction Number)**: Average number of secondary infections (R₀ = β/γ)
- **Herd Immunity Threshold**: Fraction needed to be immune to stop spread (1 - 1/R₀)
- **Attack Rate**: Total fraction of population infected during epidemic
- **Peak Infection**: Maximum fraction infected at any time point

### Interventions

- **Lockdown**: Strong reduction in contact rate (typical effectiveness: 60-80%)
- **Social Distancing**: Moderate reduction in contact rate (typical effectiveness: 30-50%)
- **Vaccination**: Reduces effective transmission by immunizing population
- **Contact Tracing**: Isolates infected individuals quickly

## Extending the Framework

### Creating a New Model

```python
from core.base_models import CompartmentalModel, ModelParameters
import numpy as np

class MyCustomModel(CompartmentalModel):
    def __init__(self, params: ModelParameters):
        super().__init__(params)
        self.state_names = ['S', 'I', 'R']  # Define compartments

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        # Implement your differential equations
        S, I, R = y
        # ... your equations here
        return np.array([dS, dI, dR])

    def get_initial_conditions(self) -> np.ndarray:
        # Return initial state
        return np.array([0.99, 0.01, 0.0])
```

### Adding Custom Interventions

```python
def my_intervention(start_time, duration, effectiveness):
    return {
        'start': start_time,
        'end': start_time + duration,
        'effectiveness': effectiveness,
        'type': 'reduction',
        'name': 'My Custom Intervention'
    }
```

## Analysis Tools

The framework provides several analysis utilities:

```python
from analysis.metrics import (
    calculate_peak_time,
    calculate_epidemic_duration,
    compare_interventions
)

from analysis.visualization import (
    plot_comparison,
    plot_phase_portrait,
    plot_R_effective
)

# Calculate metrics
peak_time, peak_value = calculate_peak_time(results, 'I')
duration = calculate_epidemic_duration(results)

# Create visualizations
plot_phase_portrait(results, 'S', 'I')
plot_R_effective(results, beta=0.5, gamma=0.1)
```

## References

- Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics.
- Anderson, R. M., & May, R. M. (1991). Infectious diseases of humans: dynamics and control.
- Keeling, M. J., & Rohani, P. (2008). Modeling infectious diseases in humans and animals.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Contact

For questions or feedback, please open an issue on the repository.
