from datetime import datetime

from base_classes.operator import AbstractOperator

class AbstractGraphNode:
    _node_id: str
    _operator_id: str
    def __init__(self, operator_id: str) -> None:
        list_of_initiated_operator = AbstractOperator.get_operator_ids()
        if operator_id in list_of_initiated_operator:
            self._operator_id = operator_id
        else:
            raise ValueError(f"Operator ID {operator_id} is not initiated.")
    @property
    def node_id(self) -> str:
        return self._node_id
    @property
    def operator_id(self) -> str:
        return self._operator_id