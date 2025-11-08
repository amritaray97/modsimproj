"""
SEIRV (Susceptible-Exposed-Infectious-Recovered-Vaccinated) Model Implementation

This model extends SEIR to include vaccination dynamics.


The compartments are defined as follows:
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

In the set_vaccination_campaign func, we set up a time-limited vaccination campaign.
"""
import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')


import numpy as np
from core.base_models import CompartmentalModel, SEIRParameters
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class SEIRVParameters(SEIRParameters):
    vaccine_efficacy: float = 0.8  # Fraction of vaccinated who become immune
    vaccination_rate: float = 0.01  # Fraction of susceptibles vaccinated per day


class SEIRVModel(CompartmentalModel):
    

    def __init__(self,
                 params: SEIRVParameters,
                 S0: float = 0.99,
                 E0: float = 0.0,
                 I0: float = 0.01,
                 R0: float = 0.0,
                 V0: float = 0.0,
                 vaccination_schedule: Optional[Callable[[float], float]] = None):
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
        
        return np.array([self.S0, self.E0, self.I0, self.R0, self.V0]) #Return initial conditions [S0, E0, I0, R0, V0]

    
# start_time: When vaccination starts
# duration: How long vaccination continues
# rate: Vaccination rate (fraction per day) during campaign
# efficacy: Vaccine efficacy (if None, uses params.vaccine_efficacy)

    def set_vaccination_campaign(self,
                                start_time: float,
                                duration: float,
                                rate: float,
                                efficacy: Optional[float] = None):
       
        if efficacy is not None:
            self.params.vaccine_efficacy = efficacy

        def campaign_schedule(t):
            if start_time <= t <= start_time + duration:
                return rate
            return 0.0

        self.vaccination_schedule = campaign_schedule



# We now calculate total number vaccinated (both effective and ineffective).
# This will return the total fraction of the population vaccinated
    def calculate_total_vaccinated(self, results: Optional[dict] = None) -> float:

        if results is None:
            results = self.results

        if 'V' in results and 'R' in results:
            # V contains failed vaccinations, R contains natural recovery + successful vaccinations
            # We need to estimate successful vaccinations
            # Final V + (vaccinated who went to R)
            return results['V'][-1] + (results['R'][-1] - self.R0)

        return 0.0


# We calculate attack rate (total infections, excluding vaccine-prevented).
# This will return the Attack rate as fraction

    def calculate_attack_rate(self, results: Optional[dict] = None) -> float:
        
        if results is None:
            results = self.results

       
        if 'R' in results and 'I' in results:
            # This is approximate - exact tracking would require additional state
            return results['R'][-1] + results['I'][-1]

        return 0.0


# We calculate number of infections prevented compared to baseline. 
# The baseline_attack_rate is defined as Attack rate without vaccination
# The results will give us the fraction of infections prevented

    def calculate_infections_prevented(self, baseline_attack_rate: float,
                                      results: Optional[dict] = None) -> float:
        
        vaccinated_attack_rate = self.calculate_attack_rate(results)
        return baseline_attack_rate - vaccinated_attack_rate



# We calculate percent reduction in infections. This gives us the percent reduction (0-100)

    def calculate_percent_reduction(self, baseline_attack_rate: float,
                                   results: Optional[dict] = None) -> float:
        infections_prevented = self.calculate_infections_prevented(baseline_attack_rate, results)
        if baseline_attack_rate > 0:
            return 100 * infections_prevented / baseline_attack_rate
        return 0.0
