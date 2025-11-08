"""
SIR (Susceptible-Infectious-Recovered) Model Implementation. This is the classic epidemic model, and sort of 
forms the base of the other models.
The compartments are defined as:
        S - Susceptible: individuals who can contract the disease
        I - Infectious: individuals who are infected and can transmit
        R - Recovered: individuals who have recovered and are immune

    Parameters:
        beta: transmission rate
        gamma: recovery rate

    Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I
"""

import numpy as np
from core.base_models import CompartmentalModel, SIRParameters


class SIRModel(CompartmentalModel):

    def __init__(self, params: SIRParameters, S0: float = 0.99, I0: float = 0.01, R0: float = 0.0):
        super().__init__(params)
        self.state_names = ['S', 'I', 'R']
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Here we calculate the derivatives for the SIR model.

        Args:
            t: Current time
            y: Current state [S, I, R]

        Returns:
            Array of derivatives [dS/dt, dI/dt, dR/dt]
        """
        
        S, I, R = y

        # Force of infection
        lambda_t = self.params.beta * I

        # Derivatives
        dS = -lambda_t * S
        dI = lambda_t * S - self.params.gamma * I
        dR = self.params.gamma * I

        return np.array([dS, dI, dR])

    def get_initial_conditions(self) -> np.ndarray:
        return np.array([self.S0, self.I0, self.R0])

    def calculate_herd_immunity_threshold(self) -> float:
        """
        Calculation of the herd immunity threshold.

        Returns:
            The fraction of population that needs to be immune
        """
        
        R0 = self.calculate_R0()
        return 1 - (1 / R0) if R0 > 1 else 0.0
