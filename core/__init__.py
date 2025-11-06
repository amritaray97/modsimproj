"""
Core module for epidemic modeling framework.

This module provides the base classes and parameters for epidemic models.
"""

from .base_models import (
    ModelParameters,
    SIRParameters,
    SEIRParameters,
    SIRSParameters,
    BaseEpidemicModel,
    CompartmentalModel
)

__all__ = [
    'ModelParameters',
    'SIRParameters',
    'SEIRParameters',
    'SIRSParameters',
    'BaseEpidemicModel',
    'CompartmentalModel'
]
