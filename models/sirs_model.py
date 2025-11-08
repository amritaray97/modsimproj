"""
SIRS (Susceptible-Infectious-Recovered-Susceptible) Model Implementation. 
This is a model that shows waning immunity. 
We define the compartments as follows:
        S - Susceptible: individuals who can contract the disease
        I - Infectious: individuals who are infected and can transmit
        R - Recovered: individuals who have recovered and have temporary immunity

    Parameters:
        beta: transmission rate
        gamma: recovery rate
        omega: rate of waning immunity (recovered -> susceptible)

    Equations:
        dS/dt = -beta * S * I + omega * R
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I - omega * R

We proceed to calculate the derivatives for the SIRS model.

Args:
    t: Current time
    y: Current state [S, I, R]

Returns:
    Array of derivatives [dS/dt, dI/dt, dR/dt]

With waning immunity, the disease can become endemic with sustained oscillations.
"""
import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')

from typing import Optional

import numpy as np
from core.base_models import CompartmentalModel, SIRSParameters


class SIRSModel(CompartmentalModel):

    def __init__(self, params: SIRSParameters, S0: float = 0.99, I0: float = 0.01, R0: float = 0.0):
        super().__init__(params)
        self.state_names = ['S', 'I', 'R']
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        
        S, I, R = y

        # Force of infection
        lambda_t = self.params.beta * I

        # Derivatives
        dS = -lambda_t * S + self.params.omega * R
        dI = lambda_t * S - self.params.gamma * I
        dR = self.params.gamma * I - self.params.omega * R

        return np.array([dS, dI, dR])

    def get_initial_conditions(self) -> np.ndarray:
        return np.array([self.S0, self.I0, self.R0])

# Now we have to calculate the endemic equilibrium if it exists. This will return a dictionary 
# with equilibrium values for S, I, R

    def calculate_endemic_equilibrium(self) -> dict:
        
        R0 = self.calculate_R0()

        if R0 <= 1:
            # Disease-free equilibrium
            return {'S': 1.0, 'I': 0.0, 'R': 0.0}

        # Endemic equilibrium
        S_star = 1.0 / R0
        I_star = self.params.omega * (1 - S_star) / (self.params.gamma + self.params.omega)
        R_star = 1 - S_star - I_star

        return {'S': S_star, 'I': I_star, 'R': R_star}


    def calculate_R_effective(self, results: Optional[dict] = None) -> np.ndarray:
    
        if results is None:
            results = self.results
        
        if 'S' not in results:
            raise ValueError("Results must contain S compartment")
        
        S = results['S']
        R_eff = self.params.R0 * S
        
        return R_eff
