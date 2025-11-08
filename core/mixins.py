import numpy as np
from typing import Dict, Tuple, List
import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')

from models.seir_model import SEIRModel

"""
Mixin for adding stochasticity to models
"""
class StochasticMixin:


    def add_noise(self, derivatives: np.ndarray,
                 state: np.ndarray,
                 noise_scale: float = 0.01,
                 dt: float = 0.01) -> np.ndarray:

        # Demographic noise for each compartment
        noise = np.random.normal(0, 1, len(state))
        # Scaled by sqrt of transition rates 
        noise *= noise_scale * np.sqrt(dt)
        return derivatives + noise

    def simulate_stochastic(self,
                           initial_conditions: np.ndarray,
                           t_span: Tuple[float, float],
                           dt: float = 0.01,
                           noise_scale: float = 0.01) -> Dict:

        t = np.arange(t_span[0], t_span[1], dt) # Simulate with stochastic noise (Euler-Maruyama)
        n_steps = len(t)
        n_states = len(initial_conditions)

        trajectory = np.zeros((n_steps, n_states))
        trajectory[0] = initial_conditions

        for i in range(1, n_steps):
            # Deterministic part
            dy = self.derivatives(t[i-1], trajectory[i-1]) * dt

            # noise step
            dy_stochastic = self.add_noise(dy, trajectory[i-1], noise_scale, dt)

            # Updated state
            trajectory[i] = trajectory[i-1] + dy_stochastic
            trajectory[i] = np.maximum(trajectory[i], 0)  # Ensure non-negative

            # Normalized to maintain constant population
            trajectory[i] = trajectory[i] / np.sum(trajectory[i]) * self.params.population

        results = {'t': t}
        for j, name in enumerate(self.state_names):
            results[name] = trajectory[:, j]

        return results


# Mixin for adding intervention capabilities

class InterventionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interventions = []

    # Intervention that modifies transmission
    def add_intervention(self,
                        start_time: float,
                        duration: float,
                        effectiveness: float,
                        intervention_type: str = "reduction"):

        self.interventions.append({
            'start': start_time,
            'end': start_time + duration,
            'effectiveness': effectiveness,
            'type': intervention_type
        })

    # Transmission rate modifier at time t
    def get_intervention_modifier(self, t: float) -> float:

        modifier = 1.0

        for intervention in self.interventions:
            if intervention['start'] <= t <= intervention['end']:
                if intervention['type'] == 'reduction':
                    modifier *= (1 - intervention['effectiveness'])
                elif intervention['type'] == 'increase':
                    modifier *= (1 + intervention['effectiveness'])

        return modifier

    def derivatives_with_intervention(self, t: float, y: np.ndarray) -> np.ndarray:
        modifier = self.get_intervention_modifier(t)
        original_beta = self.params.beta
        self.params.beta = original_beta * modifier
        result = SEIRModel.derivatives(self, t, y) 
        self.params.beta = original_beta
        return result


# Mixin for network-based epidemic models
class NetworkModelMixin:
   

    def set_network(self, network):
        self.network = network
        self.n_nodes = network.number_of_nodes()

    def get_infected_neighbors(self, node: int, states: np.ndarray) -> List[int]:
        neighbors = list(self.network.neighbors(node))
        return [n for n in neighbors if states[n] == 1]  # 1 = infected state

    def calculate_force_of_infection(self, node: int, states: np.ndarray) -> float:
        infected_neighbors = self.get_infected_neighbors(node, states)

        if not infected_neighbors:
            return 0.0

        # Can be modified to include edge weights, node attributes, etc.
        # return self.params.beta * len(infected_neighbors) / self.network.degree(node)

        # Per-contact transmission probability
        return self.params.beta * len(infected_neighbors)
