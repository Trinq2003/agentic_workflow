from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Iterable

from base_classes.llm import AbstractLanguageModel
from base_classes.tool import AbstractTool

class AbstractOperator(ABC):
    """
    Abstract base class that defines the interface for all operators.
    """
    operator_id: str = None
    _llm_component: Iterable[AbstractLanguageModel] = None
    _tool_component: Iterable[AbstractTool] = None
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the AbstractOperator instance with configuration.

        :param config: The operator configuration object.
        :type config: Dict[str, Any]
        """
        pass

    @abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the operator on the input data.

        :param data: The input data.
        :type data: Dict[str, Any]

        :return: The output data.
        :rtype: Dict[str, Any]
        """
        pass