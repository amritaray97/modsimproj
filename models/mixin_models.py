"""
Combination models using mixins for enhanced functionality.
"""

from core.mixins import StochasticMixin, InterventionMixin
from .sir_model import SIRModel
from .seir_model import SEIRModel


# SIR model with stochastic capability
class StochasticSIR(StochasticMixin, SIRModel):
    """SIR model with stochastic simulation capability"""
    pass


# SEIR model with interventions
class SEIRWithInterventions(InterventionMixin, SEIRModel):
    """SEIR model with intervention capabilities"""

    def derivatives(self, t: float, y) -> np.ndarray:
        """Use intervention-modified derivatives"""
        return self.derivatives_with_intervention(t, y)
