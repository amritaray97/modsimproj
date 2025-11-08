"""
Combination models using mixins for enhanced functionality.
"""
import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')


import numpy as np
from core.mixins import StochasticMixin, InterventionMixin
from models.sir_model import SIRModel
from models.seir_model import SEIRModel


# SIR model with stochastic capability
class StochasticSIR(StochasticMixin, SIRModel):
    pass


# SIR model with interventions
class SIRWithInterventions(InterventionMixin, SIRModel):
    """SIR model with intervention capabilities"""

    def derivatives(self, t: float, y) -> np.ndarray:
        """Use intervention-modified derivatives"""
        return self.derivatives_with_intervention(t, y)


# SEIR model with interventions
class SEIRWithInterventions(InterventionMixin, SEIRModel):
    def derivatives(self, t: float, y) -> np.ndarray:
        
        return self.derivatives_with_intervention(t, y) # intervention-modified derivatives
