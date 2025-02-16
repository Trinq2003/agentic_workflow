from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Self

from base_classes.graph.node import AbstractGraphNode

class AbstractGraphEdge(ABC):
    _start_node: AbstractGraphNode
    _end_node: AbstractGraphNode
    _edge_type: str
    _description: Optional[str]
    _edge_id: int
    _list_of_edge_ids: List[int] = None
    _edge_instances_by_id: Dict[str, Self] = None
    def __init__(self, start_node: AbstractGraphNode, end_node: AbstractGraphNode, edge_type: str, description: str = '') -> None:        
        self._start_node = start_node
        self._end_node = end_node
        self._edge_type = edge_type
        self._description = description
        
        self._edge_id = self.__class__._list_of_edge_ids[-1] + 1 if self.__class__._list_of_edge_ids else 0
        self.__class__._list_of_edge_ids.append(self._edge_id)
        
    @property
    def start_node(self) -> AbstractGraphNode:
        return self._start_node
    @property
    def end_node(self) -> AbstractGraphNode:
        return self._end_node
    @property
    def edge_type(self) -> str:
        return self._edge_type
    @property
    def description(self) -> str:
        return self._description
    
    @classmethod
    def get_edge_ids(cls) -> List[int]:
        """
        Get the list of edge IDs.

        :return: The list of edge IDs.
        :rtype: str
        """
        return cls._list_of_edge_ids
    @classmethod
    def get_edge_instance_by_id(cls, edge_id) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param edge_id: The unique identifier of the edge instance.
        :return: The instance if found, otherwise None.
        """
        return cls._edge_instances_by_id.get(edge_id, None)