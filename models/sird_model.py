"""
SIRD (Susceptible-Infectious-Recovered-Dead) Model Implementation

This model extends the classic SIR model by adding a Death compartment
to track disease-induced mortality.

Compartments:
    S - Susceptible: individuals who can contract the disease
    I - Infectious: individuals who are infected and can transmit
    R - Recovered: individuals who have recovered and are immune
    D - Dead: individuals who died from the disease

Parameters:
    beta: transmission rate (infection rate)
    gamma: recovery rate
    mu: disease-induced mortality rate (death rate)

Equations:
    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I - mu * I
    dR/dt = gamma * I
    dD/dt = mu * I

Key Features:
    - Tracks cumulative deaths over time
    - Calculates case fatality rate (CFR)
    - Computes total attack rate (infections)
"""

import numpy as np
from core.base_models import CompartmentalModel, SIRDParameters
from typing import Dict, Optional


class SIRDModel(CompartmentalModel):
    """
    SIRD epidemic model with death compartment.

    This model is useful for diseases with significant mortality,
    such as Ebola, SARS, or severe pandemic influenza strains.
    """

    def __init__(self,
                 params: SIRDParameters,
                 S0: float = 0.99,
                 I0: float = 0.01,
                 R0: float = 0.0,
                 D0: float = 0.0):
        """
        Initialize the SIRD model.

        Args:
            params: SIRDParameters object with beta, gamma, mu, and population
            S0: Initial fraction of susceptible population
            I0: Initial fraction of infectious population
            R0: Initial fraction of recovered population
            D0: Initial fraction of dead population (usually 0)
        """
        super().__init__(params)
        self.state_names = ['S', 'I', 'R', 'D']
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.D0 = D0

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate derivatives for the SIRD model.

        Args:
            t: Current time
            y: Current state [S, I, R, D]

        Returns:
            Array of derivatives [dS/dt, dI/dt, dR/dt, dD/dt]
        """
        S, I, R, D = y

        # Force of infection
        lambda_t = self.params.beta * I

        # Derivatives
        dS = -lambda_t * S
        dI = lambda_t * S - self.params.gamma * I - self.params.mu * I
        dR = self.params.gamma * I
        dD = self.params.mu * I

        return np.array([dS, dI, dR, dD])

    def get_initial_conditions(self) -> np.ndarray:
        """Return initial state vector [S0, I0, R0, D0]."""
        return np.array([self.S0, self.I0, self.R0, self.D0])

    def calculate_herd_immunity_threshold(self) -> float:
        """
        Calculate the herd immunity threshold.

        For SIRD, this is the same as SIR: 1 - 1/R0

        Returns:
            Fraction of population that needs to be immune
        """
        R0_value = self.calculate_R0()
        return 1 - (1 / R0_value) if R0_value > 1 else 0.0

    def calculate_case_fatality_rate(self, results: Optional[Dict] = None) -> float:
        """
        Calculate the Case Fatality Rate (CFR).

        CFR = Deaths / (Deaths + Recovered)

        This gives the proportion of infections that result in death.

        Args:
            results: Simulation results dictionary (uses self.results if None)

        Returns:
            Case Fatality Rate as a fraction (0-1)
        """
        if results is None:
            results = self.results

        if results is None:
            raise ValueError("No results available. Run simulate() first.")

        final_D = results['D'][-1]
        final_R = results['R'][-1]

        total_resolved = final_D + final_R

        if total_resolved == 0:
            return 0.0

        cfr = final_D / total_resolved
        return cfr

    def calculate_total_deaths(self, results: Optional[Dict] = None) -> float:
        """
        Calculate total deaths as fraction of population.

        Args:
            results: Simulation results dictionary (uses self.results if None)

        Returns:
            Total deaths as fraction of initial population
        """
        if results is None:
            results = self.results

        if results is None:
            raise ValueError("No results available. Run simulate() first.")

        return results['D'][-1]

    def calculate_mortality_rate(self, results: Optional[Dict] = None) -> float:
        """
        Calculate population mortality rate.

        This is deaths per capita (total population).

        Args:
            results: Simulation results dictionary (uses self.results if None)

        Returns:
            Mortality rate as fraction of total population
        """
        return self.calculate_total_deaths(results)

    def get_epidemic_summary(self, results: Optional[Dict] = None) -> Dict:
        """
        Get comprehensive summary statistics for the epidemic.

        Args:
            results: Simulation results dictionary (uses self.results if None)

        Returns:
            Dictionary with key epidemic metrics
        """
        if results is None:
            results = self.results

        peak_time, peak_infections = self.calculate_peak_infection(results)
        attack_rate = self.calculate_attack_rate(results)
        cfr = self.calculate_case_fatality_rate(results)
        total_deaths = self.calculate_total_deaths(results)

        summary = {
            'R0': self.calculate_R0(),
            'herd_immunity_threshold': self.calculate_herd_immunity_threshold(),
            'peak_infection_time': peak_time,
            'peak_infection_fraction': peak_infections,
            'attack_rate': attack_rate,
            'case_fatality_rate': cfr,
            'total_deaths_fraction': total_deaths,
            'total_deaths_count': total_deaths * self.params.population,
            'final_susceptible': results['S'][-1],
            'final_recovered': results['R'][-1]
        }

        return summary
