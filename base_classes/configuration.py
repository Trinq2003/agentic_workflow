from abc import ABC, abstractmethod
import re
from sympy import Q
import toml
from typing import Any, Dict

class Configuration(ABC):
    def __init__(self):
        self._properties = dict()
        properties = self._init_properties()
        for property_, value, transform_fn in properties:
            if transform_fn is not None:
                value = transform_fn(value)

            self._properties[property_] = {
                'default-value': value,
                'transform_fn': transform_fn
            }
        
        self._sensitive_properties = []  # Use a single underscore

    @property
    def sensitive_properties(self):
        return self._sensitive_properties

    @sensitive_properties.setter
    def sensitive_properties(self, value):
        self._sensitive_properties = value

    @abstractmethod
    def _init_properties(self):
        """
        Abstract method that should return a list of properties in the format
        [[name, default-value, transform_fn]].
        """
        pass
    
    def _is_sensitive(self, key: str) -> bool:
        """Define which keys are sensitive."""
        return key in self._sensitive_properties
    
    def _parse_hierarchical(self, section: str, cfg_data: Dict[str, Any]) -> None:
        """
        Recursively parse the hierarchical structure of the configuration file.
        
        :param section: The current section being processed.
        :param cfg_data: The hierarchical data from the TOML file.
        :return: None
        """
        for key, value in cfg_data.items():
            if isinstance(value, dict):
                self._parse_hierarchical(f"{section}.{key}", value)
            else:
                property_ = f"{section.lstrip('.')}.{key}" if section else key
                transform_fn = self._properties.get(property_, {}).get('transform_fn', None)

                if transform_fn is not None:
                    value = transform_fn(value)

                full_key = f"{section.lower().strip('.')}_{key}" if section else f"{key}"
                setattr(self, full_key, value)

    def _mask_secret(self, value: str, visible_chars: int = 1):
        """
        Mask the secret value, showing only the first few characters.
        
        :param value: The string to be masked.
        :param visible_chars: Number of visible characters at the start.
        :return: Masked string with visible_chars shown and the rest masked.
        """
        if len(value) <= visible_chars:
            return '*' * len(value)
        return value[:visible_chars] + '*' * (len(value) - visible_chars)

    def load(self, path):
        """
        Load the configuration from the TOML file specified by the path.
        
        :param path: Path to the TOML config file.
        :return: None
        """
        config = toml.load(path)
        self._parse_hierarchical("", config)

    def __str__(self):
        result = f"{self.__class__.__name__} configuration:\n"
        for key in self.__dict__.keys():
            if not key.startswith('_'):

                value = getattr(self, key)
                if self._is_sensitive(key):
                    value = self._mask_secret(str(value))
                result += f"{key}: {value}\n"

        return result