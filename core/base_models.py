import numpy as np
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


"""Base parameters class"""
@dataclass
class ModelParameters:
    population: float = 1.0

"""SIR model parameters"""
@dataclass
class SIRParameters(ModelParameters):
    beta: float = 0.5
    gamma: float = 0.1

    @property
    def R0(self) -> float:
        return self.beta / self.gamma


@dataclass
class SEIRParameters(SIRParameters):
    """SEIR model parameters"""
    sigma: float = 0.2

    @property
    def incubation_period(self) -> float:
        return 1.0 / self.sigma

@dataclass
class SIRSParameters(SIRParameters):
    omega: float = 0.01  # Waning immunity rate


"""Abstract base class for all epidemic models"""
class BaseEpidemicModel(ABC):
    def __init__(self, params: ModelParameters):
        self.params = params
        self.results = None
        self.state_names = []

    @abstractmethod
    def derivatives(self, t: float, y: np.ndarray, *args) -> np.ndarray:
        """Calculate derivatives - must be implemented by subclasses"""
        pass

    """
    Args:
            initial_conditions: Initial state values
            t_span: Time span for simulation
            t_eval: Times at which to evaluate solution
            method: Integration method
    """
    @abstractmethod
    def get_initial_conditions(self) -> np.ndarray:
        pass

    def simulate(self,
                 initial_conditions: Optional[np.ndarray] = None,
                 t_span: Tuple[float, float] = (0, 200),
                 t_eval: Optional[np.ndarray] = None,
                 method: str = 'RK45') -> Dict:

        if initial_conditions is None:
            initial_conditions = self.get_initial_conditions()

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        solution = solve_ivp(
            self.derivatives,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method=method,
            rtol=1e-8
        )

        self.results = {'t': solution.t}
        for i, name in enumerate(self.state_names):
            self.results[name] = solution.y[i]

        return self.results

    def plot_dynamics(self,
                     results: Optional[Dict] = None,
                     ax: Optional[plt.Axes] = None,
                     **kwargs) -> plt.Axes:
        if results is None:
            results = self.results

        if results is None:
            raise ValueError("No results to plot. Run simulate() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Default colors for common compartments
        default_colors = {
            'S': 'blue',
            'E': 'yellow',
            'I': 'red',
            'R': 'green',
            'V': 'purple',
            'D': 'black',
            'M': 'black',
            'C': 'orange'
        }

        for name in self.state_names:
            if name in results:
                color = kwargs.get(f'{name}_color', default_colors.get(name, None))
                ax.plot(results['t'], results[name],
                       label=self._get_label(name),
                       color=color, linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction of Population')
        ax.set_title(f'{self.__class__.__name__} Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def _get_label(self, compartment: str) -> str:
        """Get human-readable label for compartment"""
        labels = {
            'S': 'Susceptible',
            'E': 'Exposed',
            'I': 'Infectious',
            'R': 'Recovered',
            'V': 'Vaccinated',
            'D': 'Diseased',
            'M': 'Mortality',
            'C': 'Carriers'
        }
        return labels.get(compartment, compartment)

    def calculate_peak_infection(self, results: Optional[Dict] = None) -> Tuple[float, float]:
        if results is None:
            results = self.results

        if 'I' not in results:
            raise ValueError("No infectious compartment in results")

        peak_idx = np.argmax(results['I'])
        return results['t'][peak_idx], results['I'][peak_idx]

    def calculate_attack_rate(self, results: Optional[Dict] = None) -> float:
        if results is None:
            results = self.results

        if 'R' in results:
            return results['R'][-1]
        elif 'cumulative_infections' in results:
            return results['cumulative_infections'][-1]
        else:
            raise ValueError("Cannot calculate attack rate from results")



class CompartmentalModel(BaseEpidemicModel):
    def __init__(self, params: ModelParameters):
        super().__init__(params)

    def calculate_R0(self) -> float:
        """Calculate basic reproduction number"""
        if hasattr(self.params, 'R0'):
            return self.params.R0
        return np.nan

    def calculate_R_effective(self, results: Optional[Dict] = None) -> np.ndarray:
        if results is None:
            results = self.results

        if 'S' in results and hasattr(self.params, 'beta'):
            S = results['S']
            R_eff = self.params.beta * S / self.params.gamma
            return R_eff
        return np.array([])
