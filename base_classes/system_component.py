from typing import List, Dict, Any, Self
import logging

from base_classes.configuration import Configuration
from base_classes.logger import HasLoggerClass

class SystemComponent(HasLoggerClass):
    """Abstract base class that defines the interface for all system components."""
    _config: Configuration = None
    _component_id: str = None
    _component_instances_by_id: Dict[str, Self] = {}
    def __init__(self, **kwargs) -> None:
        """
        Initialize the SystemComponent instance with configuration.

        :param config: The system component configuration object.
        :type config: Dict[str, Any]
        """
        super().__init__()
        self._component_id = "SYSTEM_COMPONENT | " + str(int(list(self.__class__._component_instances_by_id.keys())[-1].split(" | ")[1])+1) if self._component_instances_by_id.keys() else "SYSTEM_COMPONENT | 0"
        self.logger.debug(f"System Component ID: {self._component_id} | Component type: {self.__class__}")
        if self._component_id in self.__class__._component_instances_by_id.keys():
            self.logger.error(f"System Component ID {self._component_id} is already initiated.")
            raise ValueError(f"System Component ID {self._component_id} is already initiated.")
        else:
            self.__class__._component_instances_by_id[self._component_id] = self
    
    @property
    def component_id(self) -> str:
        return self._component_id
    
    @classmethod
    def get_component_ids(cls) -> List[str]:
        """
        Get the list of system component IDs.

        :return: The list of system component IDs.
        :rtype: List[str]
        """
        return cls._component_instances_by_id.keys()
    
    @classmethod
    def get_component_instance_by_id(cls, component_id: str) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._component_instances_by_id.get(component_id, None)
    
    def __eq__(self, value: Self) -> bool:
        return self._component_id == value.component_id