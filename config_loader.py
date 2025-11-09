"""
Configuration Loader for Epidemic Simulator

This module provides functionality to load and parse configuration files
for running epidemic simulations with different parameters.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _validate_config(self):
        required_fields = ['model', 'parameters', 'initial_conditions', 'simulation']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")

        # Validate model type
        valid_models = ['SIR', 'SEIR', 'SIRS', 'SIRD', 'SEIRV']
        if self.config['model']['type'] not in valid_models:
            raise ValueError(f"Invalid model type. Must be one of: {valid_models}")

    def get_model_type(self) -> str:
        return self.config['model']['type']

    def get_parameters(self) -> Dict[str, float]:
        return self.config['parameters']

    def get_initial_conditions(self) -> Dict[str, float]:
        return self.config['initial_conditions']

    def get_simulation_settings(self) -> Dict[str, Any]:
        return self.config['simulation']

    def get_interventions(self) -> Optional[list]:
        return self.config.get('interventions', None)

    def get_vaccination_settings(self) -> Optional[Dict[str, Any]]:
        return self.config.get('vaccination', None)

    def get_output_settings(self) -> Dict[str, Any]:
        defaults = {
            'save_plots': True,
            'output_dir': 'results',
            'plot_format': 'png',
            'dpi': 300,
            'show_plots': False
        }

        output_config = self.config.get('output', {})
        defaults.update(output_config)
        return defaults

    def get_stochastic_settings(self) -> Optional[Dict[str, Any]]:
        return self.config.get('stochastic', None)

    def get_all(self) -> Dict[str, Any]:
        return self.config


def load_config(config_path: str) -> ConfigLoader:
    """
    Convenience function to load a configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)
