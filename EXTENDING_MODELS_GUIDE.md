# 📖 Guide: How to Extend Models in This Codebase

This guide shows you how to easily add new epidemic models to the simulation framework.

## 🎯 Quick Summary

To add a new model, you need to:

1. **Define parameters** (what variables your model needs)
2. **Create the model class** (implement the equations)
3. **Implement 2 required methods** (`derivatives` and `get_initial_conditions`)
4. **Register it** (add to imports and config)
5. **Create a config file** (to run it easily)

That's it! Let's walk through each step with a real example.

---

## 📚 Example: Creating a SIRD Model

We'll create a **SIRD model** (SIR with Deaths) that tracks disease mortality.

**Compartments:** S → I → R/D

**New parameter:** `mu` (mortality rate)

---

## Step 1: Define Parameters

Add your parameter class to `core/base_models.py`:

```python
@dataclass
class SIRDParameters(SIRParameters):
    mu: float = 0.01  # Disease-induced mortality rate
```

**Key points:**
- Inherit from an existing parameter class (like `SIRParameters`)
- Add only the **new** parameters your model needs
- Use descriptive names and comments

---

## Step 2: Create the Model Class

Create a new file `models/sird_model.py`:

```python
import numpy as np
from core.base_models import CompartmentalModel, SIRDParameters


class SIRDModel(CompartmentalModel):
    def __init__(self,
                 params: SIRDParameters,
                 S0: float = 0.99,
                 I0: float = 0.01,
                 R0: float = 0.0,
                 D0: float = 0.0):
        super().__init__(params)
        self.state_names = ['S', 'I', 'R', 'D']  # ← Define compartments
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.D0 = D0

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        YOUR MODEL EQUATIONS GO HERE!
        This is the most important method.
        """
        S, I, R, D = y  # Unpack state vector

        # Calculate derivatives (your ODE equations)
        dS = -self.params.beta * S * I
        dI = self.params.beta * S * I - self.params.gamma * I - self.params.mu * I
        dR = self.params.gamma * I
        dD = self.params.mu * I

        return np.array([dS, dI, dR, dD])  # Must match order of state_names!

    def get_initial_conditions(self) -> np.ndarray:
        """Return initial state vector [S0, I0, R0, D0]"""
        return np.array([self.S0, self.I0, self.R0, self.D0])
```

**Key points:**
- Inherit from `CompartmentalModel` (or `BaseEpidemicModel`)
- Set `self.state_names` - this defines your compartments
- Implement `derivatives()` - your ODE equations
- Implement `get_initial_conditions()` - return initial values
- Arrays must be in the **same order** as `state_names`!

---

## Step 3: Add Custom Analysis Methods (Optional)

You can add model-specific metrics:

```python
def calculate_case_fatality_rate(self, results=None):
    """Calculate the proportion of infections that result in death"""
    if results is None:
        results = self.results

    final_D = results['D'][-1]
    final_R = results['R'][-1]
    total_resolved = final_D + final_R

    return final_D / total_resolved if total_resolved > 0 else 0.0
```

---

## Step 4: Register Your Model

### A. Update `models/__init__.py`:

```python
from .sird_model import SIRDModel  # Add import

__all__ = [
    'SIRModel',
    'SEIRModel',
    'SIRSModel',
    'SIRDModel',  # Add to exports
    # ... other models
]
```

### B. Update `config_loader.py`:

```python
valid_models = ['SIR', 'SEIR', 'SIRS', 'SIRD', 'SEIRV']  # Add 'SIRD'
```

### C. Update `run_simulation.py`:

Add imports at the top:

```python
from core.base_models import SIRDParameters
from models.sird_model import SIRDModel
```

Add to `create_model()` method:

```python
elif model_type == 'SIRD':
    params = SIRDParameters(**params_dict)
    self.model = SIRDModel(
        params=params,
        S0=initial_conditions['S0'],
        I0=initial_conditions['I0'],
        R0=initial_conditions['R0'],
        D0=initial_conditions.get('D0', 0.0)
    )
```

---

## Step 5: Create a Configuration File

Create `configs/sird_example.json`:

```json
{
  "model": {
    "type": "SIRD",
    "description": "SIRD model - SIR with Deaths compartment"
  },
  "parameters": {
    "beta": 0.5,
    "gamma": 0.1,
    "mu": 0.02,
    "population": 1000000
  },
  "initial_conditions": {
    "S0": 0.99,
    "I0": 0.01,
    "R0": 0.0,
    "D0": 0.0
  },
  "simulation": {
    "t_start": 0,
    "t_end": 200,
    "num_points": 1000,
    "method": "RK45"
  },
  "output": {
    "save_plots": true,
    "output_dir": "results",
    "plot_format": "png",
    "dpi": 300
  }
}
```

---

## 🎉 Done! Now Use Your Model

### Method 1: Using the Config File

```bash
python run_simulation.py --config configs/sird_example.json
```

### Method 2: Programmatically

```python
from core.base_models import SIRDParameters
from models.sird_model import SIRDModel

# Create parameters
params = SIRDParameters(population=1_000_000, beta=0.5, gamma=0.1, mu=0.02)

# Create model
model = SIRDModel(params=params, S0=0.99, I0=0.01, R0=0.0, D0=0.0)

# Run simulation
results = model.simulate(t_span=(0, 200))

# Calculate metrics
peak_time, peak_infections = model.calculate_peak_infection(results)
cfr = model.calculate_case_fatality_rate(results)

# Plot
model.plot_dynamics(results=results)
```

---

## 🔧 Architecture Overview

```
Your Model Class
    ↓
CompartmentalModel (adds R0 calculations)
    ↓
BaseEpidemicModel (provides simulate(), plot_dynamics(), etc.)
```

**What you inherit for free:**
- ✅ `simulate()` - ODE solver
- ✅ `plot_dynamics()` - Automatic visualization
- ✅ `calculate_peak_infection()` - Peak detection
- ✅ `calculate_attack_rate()` - Total infections
- ✅ `calculate_R0()` - Basic reproduction number

**What you must implement:**
- ⚙️ `derivatives(t, y)` - Your ODE equations
- ⚙️ `get_initial_conditions()` - Initial state

---

## 💡 Tips & Best Practices

### 1. **Keep equations in derivatives() method**
```python
def derivatives(self, t, y):
    S, I, R = y

    # ✅ Good: Clear, readable equations
    force_of_infection = self.params.beta * I
    dS = -force_of_infection * S
    dI = force_of_infection * S - self.params.gamma * I
    dR = self.params.gamma * I

    return np.array([dS, dI, dR])
```

### 2. **Order matters!**
```python
self.state_names = ['S', 'I', 'R', 'D']  # Define order

def derivatives(self, t, y):
    S, I, R, D = y  # Unpack in SAME order
    # ...
    return np.array([dS, dI, dR, dD])  # Return in SAME order

def get_initial_conditions(self):
    return np.array([self.S0, self.I0, self.R0, self.D0])  # SAME order!
```

### 3. **Use descriptive parameter names**
```python
# ✅ Good
@dataclass
class SIRDParameters(SIRParameters):
    mu: float = 0.01  # Disease-induced mortality rate

# ❌ Bad
@dataclass
class SIRDParameters(SIRParameters):
    m: float = 0.01  # What's 'm'?
```

### 4. **Add docstrings and comments**
```python
def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
    """
    SIRD model equations.

    Equations:
        dS/dt = -β·S·I
        dI/dt = β·S·I - γ·I - μ·I
        dR/dt = γ·I
        dD/dt = μ·I

    Args:
        t: Current time
        y: State vector [S, I, R, D]

    Returns:
        Derivative vector [dS/dt, dI/dt, dR/dt, dD/dt]
    """
```

### 5. **Test your model**
```python
# Create a simple test
params = SIRDParameters(beta=0.5, gamma=0.1, mu=0.02)
model = SIRDModel(params=params)
results = model.simulate()

# Sanity checks
assert results['S'][-1] < results['S'][0]  # Susceptible should decrease
assert results['D'][-1] > 0  # Should have some deaths
assert np.isclose(
    results['S'][-1] + results['I'][-1] + results['R'][-1] + results['D'][-1],
    1.0
)  # Total should be conserved
```

---

## 🚀 Advanced: Using Mixins

Want to add interventions or stochasticity? Use mixins!

```python
from core.mixins import InterventionMixin

class SIRDWithInterventions(InterventionMixin, SIRDModel):
    def derivatives(self, t, y):
        return self.derivatives_with_intervention(t, y)

# Now you can add lockdowns, social distancing, etc.
model = SIRDWithInterventions(params=params)
model.add_intervention(
    name="Lockdown",
    start_time=30,
    duration=30,
    effectiveness=0.7,
    intervention_type="reduction"
)
```

---

## 📋 Checklist for Adding a New Model

- [ ] Define parameter class in `core/base_models.py`
- [ ] Create model file in `models/your_model.py`
- [ ] Implement `__init__`, `derivatives`, and `get_initial_conditions`
- [ ] Set `self.state_names` correctly
- [ ] Add imports to `models/__init__.py`
- [ ] Add model type to `config_loader.py` valid_models list
- [ ] Add imports and creation logic to `run_simulation.py`
- [ ] Create a config file in `configs/`
- [ ] Test your model!
- [ ] (Optional) Add custom analysis methods

---

## 🔍 Common Issues

### Issue: "Arrays have different sizes"
**Cause:** Order mismatch between `state_names`, `derivatives`, and `get_initial_conditions`

**Fix:** Ensure all three use the same order!

### Issue: "Invalid model type"
**Cause:** Forgot to add to `config_loader.py`

**Fix:** Add your model type to the `valid_models` list

### Issue: Model doesn't run
**Cause:** Missing registration in `run_simulation.py`

**Fix:** Add the `elif model_type == 'YOUR_MODEL':` block

---

## 📚 More Examples

Check out existing models for reference:

- **Simple:** `models/sir_model.py` - Basic 3-compartment model
- **Intermediate:** `models/seir_model.py` - 4 compartments with latent period
- **Advanced:** `models/seirv_model.py` - Vaccination with time-dependent rates
- **Complex:** `models/sirs_model.py` - Endemic equilibrium analysis

---

## 🎓 Understanding the Equations

Your `derivatives()` method implements the **right-hand side** of differential equations:

```
dX/dt = f(t, X)
      ↑
   This is what you return!
```

Example for SIRD:
```python
dS/dt = -β·S·I        → dS = -self.params.beta * S * I
dI/dt = β·S·I - γ·I - μ·I  → dI = self.params.beta * S * I - self.params.gamma * I - self.params.mu * I
dR/dt = γ·I           → dR = self.params.gamma * I
dD/dt = μ·I           → dD = self.params.mu * I
```

The ODE solver (`scipy.integrate.solve_ivp`) handles the integration for you!

---

## 💬 Need Help?

1. Look at existing models in `models/` for examples
2. Check the base classes in `core/base_models.py`
3. See the example usage in `example_sird_usage.py`

Happy modeling! 🎉
