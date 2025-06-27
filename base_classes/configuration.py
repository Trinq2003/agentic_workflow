from abc import abstractmethod
import yaml
from typing import Any, Dict, List
import logging

from base_classes.logger import HasLoggerClass

class Configuration(HasLoggerClass):
    _properties: Dict[str, Dict[str, Any]]
    _sensitive_properties: List[str]
    def __init__(self):
        super().__init__()
        self._properties = dict()
        properties = self._init_properties()
        for property_, value, transform_fn in properties:
            if transform_fn is not None:
                value = transform_fn(value)

            self._properties[property_] = {
                'default-value': value,
                'transform_fn': transform_fn
            }
        
        self._sensitive_properties: List[str] = []  # Define sensitive properties
        self.logger.debug(f"List of {self.__class__.__name__}'s properties: {self._properties.keys()}")
        self.logger.debug(f"List of {self.__class__.__name__}'s sensitive properties: {self._sensitive_properties}")

    @property
    def sensitive_properties(self) -> List[str]:
        return self._sensitive_properties

    @sensitive_properties.setter
    def sensitive_properties(self, value: List[str]):
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
        Recursively parse the hierarchical structure of the YAML configuration file.
        
        :param section: The current section being processed.
        :param cfg_data: The hierarchical data from the YAML file.
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

                # Convert dots to underscores in the section and key
                section_clean = section.lower().strip('.').replace('.', '_')
                full_key = f"{section_clean}_{key}" if section_clean else f"{key}"
                setattr(self, full_key, value)

    def _mask_secret(self, value: str, visible_chars: int = 1) -> str:
        """
        Mask the secret value, showing only the first few characters.
        
        :param value: The string to be masked.
        :param visible_chars: Number of visible characters at the start.
        :return: Masked string with visible_chars shown and the rest masked.
        """
        if len(value) <= visible_chars:
            return '*' * len(value)
        return value[:visible_chars] + '*' * (len(value) - visible_chars)

    def load(self, path: str) -> None:
        """
        Load the configuration from the YAML file specified by the path.
        
        :param path: Path to the YAML config file.
        :return: None
        """
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            self._parse_hierarchical("", config)

    def __str__(self) -> str:
        result = f"{self.__class__.__name__} configuration:\n"
        for key in self.__dict__.keys():
            if not key.startswith('_'):
                value = getattr(self, key)
                if self._is_sensitive(key):
                    value = self._mask_secret(str(value))
                result += f"{key}: {value}\n"

        return result
