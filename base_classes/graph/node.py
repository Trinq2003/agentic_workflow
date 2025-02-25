from abc import ABC, abstractmethod
from typing import Any, List, Dict, Self

from base_classes.memory.memory import AbstractMemory
from base_classes.system_component import SystemComponent
from base_classes.prompt import AbstractPrompt

class AbstractGraphNode(ABC):
    _node_id: int
    _list_of_node_ids: List[int] = []
    _node_instances_by_id: Dict[str, Self] = []
    def __init__(self,  **kwargs) -> None:
        self._node_id = self.__class__._list_of_node_ids[-1] + 1 if self.__class__._list_of_node_ids else 0
        self.__class__._list_of_node_ids.append(self._node_id)
        self.__class__._node_instances_by_id[self._node_id] = self

    @property
    def node_id(self) -> int:
        return self._node_id

    @classmethod
    def get_node_ids(cls) -> List[int]:
        """
        Get the list of node IDs.

        :return: The list of node IDs.
        :rtype: str
        """
        return cls._list_of_node_ids
    
    @classmethod
    def get_node_instance_by_id(cls, node_id) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param node_id: The unique identifier of the node instance.
        :return: The instance if found, otherwise None.
        """
        return cls._node_instances_by_id.get(node_id, None)
    
    @abstractmethod
    def execute(self, input: Any, **kwargs) -> Any:
        pass
    
class SystemComponentGraphNode(AbstractGraphNode):
    system_component: SystemComponent
    def __init__(self, system_component: SystemComponent, **kwargs) -> None:
        super().__init__(**kwargs)
        self.system_component = system_component
    
class MemoryRequiredGraphNode(SystemComponentGraphNode):
    memory: AbstractMemory
    def __init__(self, system_component: SystemComponent, memory: AbstractMemory, **kwargs) -> None:
        super().__init__(system_component=system_component, **kwargs)
        self.memory = memory