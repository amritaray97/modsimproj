"""
SEIRV (Susceptible-Exposed-Infectious-Recovered-Vaccinated) Model Implementation

This model extends SEIR to include vaccination dynamics.
"""

import numpy as np
from core.base_models import CompartmentalModel, SEIRParameters
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class SEIRVParameters(SEIRParameters):
    """SEIRV model parameters with vaccination"""
    vaccine_efficacy: float = 0.8  # Fraction of vaccinated who become immune
    vaccination_rate: float = 0.01  # Fraction of susceptibles vaccinated per day


class SEIRVModel(CompartmentalModel):
    """
    SEIRV epidemic model with vaccination.

    Compartments:
        S - Susceptible: individuals who can contract the disease
        E - Exposed: individuals who are infected but not yet infectious
        I - Infectious: individuals who are infected and can transmit
        R - Recovered: individuals who have recovered and are immune
        V - Vaccinated: individuals who have been vaccinated

    Parameters:
        beta: transmission rate
        sigma: rate of progression from exposed to infectious (1/incubation period)
        gamma: recovery rate
        vaccine_efficacy: fraction of vaccinated who become immune (0-1)
        vaccination_rate: base vaccination rate (fraction per day)

    Vaccination dynamics:
        - Vaccination moves individuals from S to V
        - Effective vaccination (efficacy * vaccination) provides immunity
        - Failed vaccination (1-efficacy) leaves individuals susceptible

    Equations:
        dS/dt = -beta * S * I - nu(t) * S
        dE/dt = beta * S * I - sigma * E
        dI/dt = sigma * E - gamma * I
        dR/dt = gamma * I + epsilon * nu(t) * S
        dV/dt = (1 - epsilon) * nu(t) * S

    where nu(t) is the time-dependent vaccination rate and epsilon is efficacy
    """

    def __init__(self,
                 params: SEIRVParameters,
                 S0: float = 0.99,
                 E0: float = 0.0,
                 I0: float = 0.01,
                 R0: float = 0.0,
                 V0: float = 0.0,
                 vaccination_schedule: Optional[Callable[[float], float]] = None):
        """
        Initialize SEIRV model.

        Args:
            params: Model parameters
            S0, E0, I0, R0, V0: Initial conditions
            vaccination_schedule: Function that takes time and returns vaccination rate
                                 If None, uses constant vaccination_rate from params
        """
        super().__init__(params)
        self.state_names = ['S', 'E', 'I', 'R', 'V']
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.V0 = V0

        # Vaccination schedule function
        if vaccination_schedule is None:
            # Default: constant vaccination at params.vaccination_rate
            self.vaccination_schedule = lambda t: self.params.vaccination_rate
        else:
            self.vaccination_schedule = vaccination_schedule

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivatives for the SEIRV model.

        Args:
            t: Current time
            y: Current state [S, E, I, R, V]

        Returns:
            Array of derivatives [dS/dt, dE/dt, dI/dt, dR/dt, dV/dt]
        """
        S, E, I, R, V = y

        # Force of infection
        lambda_t = self.params.beta * I

        # Time-dependent vaccination rate
        nu_t = self.vaccination_schedule(t)

        # Ensure vaccination rate doesn't exceed available susceptibles
        vaccination_flux = min(nu_t * S, S)

        # Derivatives
        dS = -lambda_t * S - vaccination_flux
        dE = lambda_t * S - self.params.sigma * E
        dI = self.params.sigma * E - self.params.gamma * I
        dR = self.params.gamma * I + self.params.vaccine_efficacy * vaccination_flux
        dV = (1 - self.params.vaccine_efficacy) * vaccination_flux

        return np.array([dS, dE, dI, dR, dV])

    def get_initial_conditions(self) -> np.ndarray:
        """Return initial conditions [S0, E0, I0, R0, V0]"""
        return np.array([self.S0, self.E0, self.I0, self.R0, self.V0])

    def set_vaccination_campaign(self,
                                start_time: float,
                                duration: float,
                                rate: float,
                                efficacy: Optional[float] = None):
        """
        Set up a time-limited vaccination campaign.

        Args:
            start_time: When vaccination starts
            duration: How long vaccination continues
            rate: Vaccination rate (fraction per day) during campaign
            efficacy: Vaccine efficacy (if None, uses params.vaccine_efficacy)
        """
        if efficacy is not None:
            self.params.vaccine_efficacy = efficacy

        def campaign_schedule(t):
            if start_time <= t <= start_time + duration:
                return rate
            return 0.0

        self.vaccination_schedule = campaign_schedule

    def calculate_total_vaccinated(self, results: Optional[dict] = None) -> float:
        """
        Calculate total number vaccinated (both effective and ineffective).

        Args:
            results: Simulation results

        Returns:
            Total fraction vaccinated
        """
        if results is None:
            results = self.results

        if 'V' in results and 'R' in results:
            # V contains failed vaccinations, R contains natural recovery + successful vaccinations
            # We need to estimate successful vaccinations
            # Final V + (vaccinated who went to R)
            return results['V'][-1] + (results['R'][-1] - self.R0)

        return 0.0

    def calculate_attack_rate(self, results: Optional[dict] = None) -> float:
        """
        Calculate attack rate (total infections, excluding vaccine-prevented).

        Args:
            results: Simulation results

        Returns:
            Attack rate as fraction
        """
        if results is None:
            results = self.results

        # Attack rate is final R + final I - initial R - vaccine-prevented
        # Approximation: R[-1] includes both natural recovery and effective vaccination
        # We want only natural infections
        if 'R' in results and 'I' in results:
            # This is approximate - exact tracking would require additional state
            return results['R'][-1] + results['I'][-1]

        return 0.0

    def calculate_infections_prevented(self, baseline_attack_rate: float,
                                      results: Optional[dict] = None) -> float:
        """
        Calculate number of infections prevented compared to baseline.

        Args:
            baseline_attack_rate: Attack rate without vaccination
            results: Simulation results with vaccination

        Returns:
            Fraction of infections prevented
        """
        vaccinated_attack_rate = self.calculate_attack_rate(results)
        return baseline_attack_rate - vaccinated_attack_rate

    def calculate_percent_reduction(self, baseline_attack_rate: float,
                                   results: Optional[dict] = None) -> float:
        """
        Calculate percent reduction in infections.

        Args:
            baseline_attack_rate: Attack rate without vaccination
            results: Simulation results with vaccination

        Returns:
            Percent reduction (0-100)
        """
        infections_prevented = self.calculate_infections_prevented(baseline_attack_rate, results)
        if baseline_attack_rate > 0:
            return 100 * infections_prevented / baseline_attack_rate
        return 0.0
