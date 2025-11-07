# Configuration Files for Epidemic Simulator

This directory contains JSON configuration files for running epidemic simulations with different parameters and scenarios.

## Quick Start

Run a simulation using any configuration file:

```bash
python run_simulation.py --config configs/sir_basic.json
```

## Available Configuration Files

### 1. `sir_basic.json`
Basic SIR model simulation with standard parameters.

**Parameters:**
- β = 0.5 (transmission rate)
- γ = 0.1 (recovery rate)
- R₀ = 5.0

**Use case:** Understanding basic epidemic dynamics without interventions.

### 2. `seir_with_interventions.json`
SEIR model with an early lockdown intervention.

**Parameters:**
- β = 0.6, σ = 0.2, γ = 0.1
- R₀ = 6.0
- Lockdown: day 30-60, 70% effective

**Use case:** Evaluating the impact of early intervention strategies.

### 3. `sirs_endemic.json`
SIRS model demonstrating endemic disease behavior with waning immunity.

**Parameters:**
- β = 0.5, γ = 0.1, ω = 0.01
- R₀ = 5.0
- Long simulation (1000 days)

**Use case:** Studying diseases with temporary immunity and endemic equilibrium.

### 4. `seirv_vaccination.json`
SEIRV model with vaccination campaign.

**Parameters:**
- β = 0.6, σ = 0.2, γ = 0.1
- Vaccine efficacy: 85%
- Vaccination: day 20-120, rate 2% per day

**Use case:** Optimizing vaccination timing and coverage.

### 5. `seir_stochastic.json`
SEIR model with stochastic noise to represent uncertainty.

**Parameters:**
- β = 0.6, σ = 0.2, γ = 0.1
- 5 realizations with 5% noise level

**Use case:** Accounting for random variability in epidemic progression.

## Configuration File Structure

A configuration file consists of several sections:

### 1. Model Section
```json
{
  "model": {
    "type": "SIR|SEIR|SIRS|SEIRV",
    "description": "Brief description of the simulation"
  }
}
```

**Supported model types:**
- `SIR`: Susceptible-Infectious-Recovered
- `SEIR`: Susceptible-Exposed-Infectious-Recovered
- `SIRS`: SIR with waning immunity
- `SEIRV`: SEIR with vaccination

### 2. Parameters Section
```json
{
  "parameters": {
    "beta": 0.5,           // Transmission rate (required for all models)
    "gamma": 0.1,          // Recovery rate (required for all models)
    "sigma": 0.2,          // Incubation rate (SEIR, SEIRV only)
    "omega": 0.01,         // Waning immunity rate (SIRS only)
    "vaccine_efficacy": 0.8,      // SEIRV only
    "vaccination_rate": 0.01,     // SEIRV only
    "population": 1.0      // Total population (normalized to 1.0)
  }
}
```

**Key parameters:**
- **β (beta)**: Contact rate × transmission probability per contact
- **γ (gamma)**: 1/infectious_period (e.g., γ=0.1 → 10 day infectious period)
- **σ (sigma)**: 1/incubation_period (e.g., σ=0.2 → 5 day incubation)
- **ω (omega)**: 1/immunity_duration (e.g., ω=0.01 → 100 day immunity)
- **R₀ (basic reproduction number)**: Automatically calculated as β/γ

### 3. Initial Conditions Section
```json
{
  "initial_conditions": {
    "S0": 0.99,    // Initial fraction susceptible
    "E0": 0.0,     // Initial fraction exposed (SEIR, SEIRV)
    "I0": 0.01,    // Initial fraction infectious
    "R0": 0.0,     // Initial fraction recovered
    "V0": 0.0      // Initial fraction vaccinated (SEIRV)
  }
}
```

**Note:** Values should sum to 1.0 (or close to 1.0).

### 4. Simulation Section
```json
{
  "simulation": {
    "t_start": 0,         // Start time
    "t_end": 200,         // End time
    "num_points": 1000,   // Number of time points
    "method": "RK45"      // Integration method (RK45, RK23, DOP853, etc.)
  }
}
```

### 5. Interventions Section (Optional)
```json
{
  "interventions": [
    {
      "name": "Lockdown",
      "start_time": 30,
      "duration": 30,
      "effectiveness": 0.7,
      "intervention_type": "reduction"
    }
  ]
}
```

**Intervention parameters:**
- **start_time**: Day when intervention begins
- **duration**: How many days intervention lasts
- **effectiveness**: Fraction reduction in transmission (0-1)
- **intervention_type**: Currently supports "reduction"

**Multiple interventions:** Add multiple objects to the array for sequential or overlapping interventions.

### 6. Vaccination Section (Optional, SEIRV only)
```json
{
  "vaccination": {
    "campaign_type": "timed",
    "start_time": 20,
    "duration": 100,
    "rate": 0.02,
    "efficacy": 0.85
  }
}
```

**Vaccination parameters:**
- **campaign_type**: "timed" for time-limited campaigns
- **start_time**: Day when vaccination starts
- **duration**: How many days campaign lasts
- **rate**: Fraction of susceptibles vaccinated per day
- **efficacy**: Vaccine effectiveness (0-1)

### 7. Stochastic Section (Optional)
```json
{
  "stochastic": {
    "enabled": true,
    "noise_level": 0.05,
    "num_realizations": 5,
    "seed": 42
  }
}
```

**Stochastic parameters:**
- **enabled**: Set to true to enable stochastic simulation
- **noise_level**: Standard deviation of Gaussian noise
- **num_realizations**: Number of stochastic runs
- **seed**: Random seed for reproducibility (optional)

### 8. Output Section (Optional)
```json
{
  "output": {
    "save_plots": true,
    "output_dir": "results",
    "plot_format": "png",
    "dpi": 300,
    "show_plots": false
  }
}
```

**Output parameters:**
- **save_plots**: Whether to save plots
- **output_dir**: Directory for saving outputs
- **plot_format**: File format (png, pdf, svg, etc.)
- **dpi**: Resolution for raster formats
- **show_plots**: Whether to display plots interactively

## Creating Custom Configurations

### Example: Custom COVID-19 Simulation

```json
{
  "model": {
    "type": "SEIR",
    "description": "COVID-19 simulation with social distancing"
  },
  "parameters": {
    "beta": 0.4,
    "sigma": 0.2,
    "gamma": 0.1,
    "population": 1.0
  },
  "initial_conditions": {
    "S0": 0.999,
    "E0": 0.0,
    "I0": 0.001,
    "R0": 0.0
  },
  "simulation": {
    "t_start": 0,
    "t_end": 365,
    "num_points": 1000,
    "method": "RK45"
  },
  "interventions": [
    {
      "name": "Social Distancing Phase 1",
      "start_time": 15,
      "duration": 60,
      "effectiveness": 0.5,
      "intervention_type": "reduction"
    },
    {
      "name": "Social Distancing Phase 2",
      "start_time": 90,
      "duration": 90,
      "effectiveness": 0.3,
      "intervention_type": "reduction"
    }
  ],
  "output": {
    "save_plots": true,
    "output_dir": "results",
    "plot_format": "png",
    "dpi": 300,
    "show_plots": false
  }
}
```

## Parameter Selection Guidelines

### Choosing β (Transmission Rate)

R₀ = β/γ, so β = R₀ × γ

**Examples:**
- Measles (R₀ ≈ 15, γ = 0.1): β ≈ 1.5
- COVID-19 (R₀ ≈ 3, γ = 0.1): β ≈ 0.3
- Seasonal flu (R₀ ≈ 1.3, γ = 0.2): β ≈ 0.26

### Choosing γ (Recovery Rate)

γ = 1 / infectious_period

**Examples:**
- 5-day infectious period: γ = 0.2
- 10-day infectious period: γ = 0.1
- 14-day infectious period: γ ≈ 0.071

### Choosing σ (Incubation Rate)

σ = 1 / incubation_period

**Examples:**
- 3-day incubation: σ ≈ 0.33
- 5-day incubation: σ = 0.2
- 7-day incubation: σ ≈ 0.14

### Choosing ω (Waning Immunity Rate)

ω = 1 / immunity_duration

**Examples:**
- 6-month immunity: ω ≈ 0.0055
- 1-year immunity: ω ≈ 0.0027
- 3-month immunity: ω ≈ 0.011

## Common Scenarios

### Scenario 1: Baseline Epidemic
- No interventions
- Use to establish baseline for comparison

### Scenario 2: Early vs. Late Intervention
- Create two configs with different intervention start_time
- Compare outcomes

### Scenario 3: Vaccination Timing
- Vary vaccination start_time
- Compare attack rates

### Scenario 4: Endemic vs. Epidemic
- Use SIRS with appropriate ω value
- Long simulation time (>1000 days)

### Scenario 5: Uncertainty Analysis
- Enable stochastic simulation
- Multiple realizations with varying parameters

## Tips

1. **Start simple**: Begin with basic configurations and add complexity gradually
2. **Validate parameters**: Ensure R₀ matches expected disease characteristics
3. **Check units**: All rates are per day, all fractions are 0-1
4. **Use meaningful names**: Include disease/scenario in description
5. **Save configs**: Keep configurations for reproducible research
6. **Version control**: Track configuration changes with git

## References

For more information on parameter estimation and model validation, see:
- Anderson & May (1991): Infectious Diseases of Humans
- Keeling & Rohani (2008): Modeling Infectious Diseases
- Project README.md for model equations

## Support

For questions or issues with configurations:
1. Check configuration file syntax (JSON validator)
2. Verify all required fields are present
3. Ensure parameter values are in valid ranges
4. Review error messages from run_simulation.py
