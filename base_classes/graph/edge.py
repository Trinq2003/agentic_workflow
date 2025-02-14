from abc import ABC, abstractmethod

from base_classes.graph.node import AbstractGraphNode

class AbstractGraphEdge:
    _start_node: AbstractGraphNode
    _end_node: AbstractGraphNode
    _edge_type: str
    def __init__(self, start_node: AbstractGraphNode, end_node: AbstractGraphNode, edge_type: str) -> None:        
        self._start_node = start_node
        self._end_node = end_node
        self._edge_type = edge_type
    @property
    def start_node(self) -> AbstractGraphNode:
        return self._start_node
    @property
    def end_node(self) -> AbstractGraphNode:
        return self._end_node
    @property
    def edge_type(self) -> str:
        return self._edge_type