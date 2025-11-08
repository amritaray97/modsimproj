# Epidemic Simulator

A comprehensive Python framework for simulating epidemic dynamics using compartmental models.

## Features

- **Multiple Epidemic Models**: SIR, SEIR, SIRS, SEIRV with extensible architecture
- **Configuration-Based Execution**: Run simulations with JSON configuration files for easy parameter management
- **Vaccination Modeling**: Time-dependent vaccination campaigns with efficacy
- **Intervention Modeling**: Simulate lockdowns, social distancing, and other NPIs
- **Stochastic Simulations**: Add noise to capture uncertainty and random effects
- **Research-Ready**: Pre-built experiments for vaccination timing optimization
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
│   ├── seirv_model.py         # SEIR with Vaccination
│   └── mixin_models.py        # Models with enhanced features
│
├── interventions/             # Intervention strategies
│   └── strategies.py          # Lockdown, social distancing, etc.
│
├── analysis/                  # Analysis and visualization tools
│   ├── metrics.py             # Analysis metrics
│   └── visualization.py       # Plotting utilities
│
├── configs/                   # Configuration files for simulations
│   ├── README.md              # Configuration documentation
│   ├── sir_basic.json         # Basic SIR configuration
│   ├── seir_with_interventions.json
│   ├── sirs_endemic.json
│   ├── seirv_vaccination.json
│   └── seir_stochastic.json
│
├── experiments/               # Example simulations
│   ├── basic_sir_simulation.py
│   ├── seir_with_interventions.py
│   ├── model_comparison.py
│   └── stochastic_simulation.py
│
├── config_loader.py           # Configuration file loader
├── run_simulation.py          # Main simulation runner
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

### Using Configuration Files (Recommended)

The easiest way to run simulations is using the configuration system:

```bash
# Run a basic SIR simulation
python run_simulation.py --config configs/sir_basic.json

# Run SEIR with interventions
python run_simulation.py --config configs/seir_with_interventions.json

# Run SEIRV with vaccination
python run_simulation.py --config configs/seirv_vaccination.json

# Run stochastic simulation
python run_simulation.py --config configs/seir_stochastic.json
```

**Available configuration templates:**
- `configs/sir_basic.json` - Basic SIR model
- `configs/seir_with_interventions.json` - SEIR with lockdown
- `configs/sirs_endemic.json` - SIRS with waning immunity
- `configs/seirv_vaccination.json` - SEIRV with vaccination campaign
- `configs/seir_stochastic.json` - SEIR with stochastic noise

See `configs/README.md` for detailed configuration documentation.

## Running Scripts

This section provides instructions for running all major scripts in the project.

### 1. Reproducibility Script

The reproducibility script runs ALL project experiments and generates all results in approximately 20-30 minutes.

```bash
# Run all experiments and generate all results
python reproduce_all_results.py
```

**What it does:**
- Runs mathematical verification of all models
- Executes all configuration-based simulations
- Generates model comparison figures
- Runs RQ1 vaccination timing analysis (quick version)
- Creates all report figures
- Produces a summary report

**Outputs:**
- `results/comprehensive_analysis/` - Verification results and visualizations
- `results/` - All simulation plots
- `results/rq1_vaccination_timing/` - Research analysis results
- `results/reproducibility_summary.json` - Complete run summary

**Runtime:** ~20-30 minutes (depending on system)

### 2. Verification Script

The verification script validates mathematical correctness and generates comprehensive visualizations.

```bash
# Verify all models and create visualizations
python verify_and_visualize.py
```

**What it does:**
- Verifies SIR, SEIR, SIRS, and SEIRV model equations
- Checks mass conservation, R₀ calculations, and equilibrium states
- Generates publication-quality visualizations:
  - Model comparison plots
  - Phase portraits for all models
  - R_effective analysis
  - Metrics summary table

**Outputs:**
- `results/comprehensive_analysis/verification_results.json` - Verification data
- `results/comprehensive_analysis/comprehensive_model_comparison.png`
- `results/comprehensive_analysis/phase_portraits.png`
- `results/comprehensive_analysis/reff_analysis.png`
- `results/comprehensive_analysis/metrics_summary.png`

**Runtime:** ~2-5 minutes

### 3. JSON Configuration-Based Simulations

Run individual simulations using JSON configuration files.

```bash
# Run a specific simulation configuration
python run_simulation.py --config configs/sir_basic.json

# Run SEIR with interventions
python run_simulation.py --config configs/seir_with_interventions.json

# Run SEIRV with vaccination
python run_simulation.py --config configs/seirv_vaccination.json

# Run stochastic simulation
python run_simulation.py --config configs/seir_stochastic.json

# Run SIRS endemic model
python run_simulation.py --config configs/sirs_endemic.json
```

**Available configurations:**
- `configs/sir_basic.json` - Basic SIR model (R₀=5.0)
- `configs/seir_with_interventions.json` - SEIR with lockdown intervention
- `configs/sirs_endemic.json` - SIRS with waning immunity
- `configs/seirv_vaccination.json` - SEIRV with vaccination campaign
- `configs/seir_stochastic.json` - SEIR with stochastic noise

**Outputs:**
- Plots saved to `results/` directory
- Named as `{config_name}_{model_type}.png`

**Runtime:** ~10-30 seconds per simulation

### 4. Analysis Scripts

The project includes specialized analysis and visualization modules.

#### Visualization Module

Use the visualization utilities programmatically:

```python
from analysis.visualization import (
    plot_comparison,      # Compare scenarios
    plot_phase_portrait,  # Create phase portraits
    plot_R_effective,     # Plot effective reproduction number
    plot_multi_compartment  # Plot multiple compartments
)

# Example: Compare two simulation results
from analysis.visualization import plot_comparison

fig = plot_comparison(
    results_list=[results1, results2],
    labels=['Scenario 1', 'Scenario 2'],
    compartment='I',
    title='Infection Comparison'
)
fig.savefig('comparison.png')
```

#### Metrics Module

Calculate epidemic metrics:

```python
from analysis.metrics import (
    calculate_peak_time,      # Find peak infection time and value
    calculate_attack_rate,    # Calculate final attack rate
    calculate_epidemic_duration  # Calculate epidemic duration
)

# Example: Calculate peak infection
from analysis.metrics import calculate_peak_time

peak_time, peak_value = calculate_peak_time(results, 'I')
print(f"Peak infections: {peak_value:.2%} at day {peak_time:.1f}")
```

#### RQ1 Vaccination Timing Research

Run the vaccination timing experiments:

```bash
# Quick version (~5-10 minutes)
python experiments/rq1_vaccination_timing_quick.py

# Full analysis (~2-3 hours)
python experiments/rq1_vaccination_timing.py
```

**What it does:**
- Analyzes optimal vaccination timing across different R₀ regimes
- Runs deterministic and stochastic simulations
- Performs sensitivity analysis on vaccine parameters
- Generates comprehensive publication-quality figures

**Outputs:**
- `results/rq1_vaccination_timing/` directory with all figures and data

See `experiments/RQ1_RESEARCH_GUIDE.md` for detailed research methodology.

#### Other Experiment Scripts

```bash
# Basic SIR simulation example
python experiments/basic_sir_simulation.py

# SEIR with interventions example
python experiments/seir_with_interventions.py

# Model comparison
python experiments/model_comparison.py

# Stochastic simulation example
python experiments/stochastic_simulation.py

# Generate report figures
python experiments/generate_report_figures.py
```

### Quick Reference Table

| Script | Command | Runtime | Purpose |
|--------|---------|---------|---------|
| **Reproducibility** | `python reproduce_all_results.py` | 20-30 min | Run ALL experiments |
| **Verification** | `python verify_and_visualize.py` | 2-5 min | Verify models & visualize |
| **JSON Simulations** | `python run_simulation.py --config configs/{file}.json` | 10-30 sec | Run single simulation |
| **RQ1 Quick** | `python experiments/rq1_vaccination_timing_quick.py` | 5-10 min | Quick vaccination analysis |
| **RQ1 Full** | `python experiments/rq1_vaccination_timing.py` | 2-3 hours | Complete vaccination research |

### Programmatic Usage

#### Basic SIR Simulation

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

#### SEIR with Interventions

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

## Research Applications

### RQ1: Vaccination Timing Optimization

A comprehensive research study investigating how vaccination campaign timing affects epidemic outcomes across different R₀ regimes.

**Quick demo:**
```bash
python experiments/rq1_vaccination_timing_quick.py
```

**Full analysis:**
```bash
python experiments/rq1_vaccination_timing.py
```

**Research questions addressed:**
- What is the optimal time to start vaccination campaigns?
- How does R₀ affect timing flexibility?
- How do vaccine efficacy and coverage rate modify optimal strategies?

See `experiments/RQ1_RESEARCH_GUIDE.md` for detailed methodology and interpretation.

**Key findings preview:**
- High R₀ diseases require early vaccination with narrow optimal windows
- Low R₀ diseases allow more timing flexibility
- Vaccine efficacy has stronger impact than timing for moderate R₀ values

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
