"""
Interventions module for epidemic control strategies.
"""

from .strategies import (
    lockdown,
    vaccination_campaign,
    social_distancing,
    contact_tracing
)

__all__ = [
    'lockdown',
    'vaccination_campaign',
    'social_distancing',
    'contact_tracing'
]
