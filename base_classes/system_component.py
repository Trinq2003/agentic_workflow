from abc import ABC, abstractmethod
from typing import List, Dict, Any, Self

from base_classes.configuration import Configuration

class SystemComponent(ABC):
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
        
        self._component_id = "SYSTEM_COMPONENT | " + str(int(self._list_of_component_ids[-1].split(" | ")[1])) if self._list_of_component_ids else "SYSTEM_COMPONENT | 0"
        
        if self._component_id in self.__class__._component_instances_by_id.keys():
            raise ValueError(f"❌ System Component ID {self._component_id} is already initiated.")
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