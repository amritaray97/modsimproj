from core.mixin import StochasticMixin, NetworkModeMixing, InterventionMixin


# SIR model with stochastic capability
class StochasticSIR(StochasticMixin, SIRModel):
    """SIR model with stochastic simulation capability"""
    pass


# SEIR model with interventions
class SEIRWithInterventions(InterventionMixin, SEIRModel):
    """SEIR model with intervention capabilities"""
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Use intervention-modified derivatives"""
        return self.derivatives_with_intervention(t, y)