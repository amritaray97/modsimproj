"""
SEIR (Susceptible-Exposed-Infectious-Recovered) Model Implementation
The compartments are defined as:
        S - Susceptible: individuals who can contract the disease
        E - Exposed: individuals who are infected but not yet infectious. This has a latent period. 
        I - Infectious: individuals who are infected and can transmit
        R - Recovered: individuals who have recovered and are immune

Parameters:
        beta: transmission rate
        sigma: rate of progression from exposed to infectious (1/incubation period)
        gamma: recovery rate

Equations:
        dS/dt = -beta * S * I
        dE/dt = beta * S * I - sigma * E
        dI/dt = sigma * E - gamma * I
        dR/dt = gamma * I
"""

import numpy as np
from core.base_models import CompartmentalModel, SEIRParameters


class SEIRModel(CompartmentalModel):
    def __init__(self, params: SEIRParameters, S0: float = 0.99, E0: float = 0.0,
                 I0: float = 0.01, R0: float = 0.0):
        super().__init__(params)
        self.state_names = ['S', 'E', 'I', 'R']
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0


"""
We now calculate the derivatives for the SEIR model.

    Args:
            t: Current time
            y: Current state [S, E, I, R]

    Returns:
            Array of derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
"""
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        
        S, E, I, R = y

        # Force of infection
        lambda_t = self.params.beta * I

        # Derivatives
        dS = -lambda_t * S
        dE = lambda_t * S - self.params.sigma * E
        dI = self.params.sigma * E - self.params.gamma * I
        dR = self.params.gamma * I

        return np.array([dS, dE, dI, dR])

    def get_initial_conditions(self) -> np.ndarray:
        return np.array([self.S0, self.E0, self.I0, self.R0])

"""
Here we calculate the herd immunity threshold.

Returns:
        The fraction of population that needs to be immune
"""
    def calculate_herd_immunity_threshold(self) -> float:
        
        R0 = self.calculate_R0()
        return 1 - (1 / R0) if R0 > 1 else 0.0
