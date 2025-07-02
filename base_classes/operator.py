from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Iterable, Self, Tuple
import uuid

from base_classes.prompt import AbstractPrompt
from base_classes.llm import AbstractLanguageModel
from base_classes.tool import AbstractTool
from base_classes.system_component import SystemComponent
from configuration.operator_configuration import OperatorConfiguration

class AbstractOperator(SystemComponent):
    """
    Abstract base class that defines the interface for all operators.
    """
    _config: OperatorConfiguration = None
    _operator_id: str = None
    _operator_type: str = None
    _enabled: bool = None
    _operator_instances_by_id: Dict[str, Self] = {}
    
    _llm_component: List[AbstractLanguageModel] = None
    _tool_component: List[AbstractTool] = None
    _execution_timeout: int = None
    _execution_max_retry: int = None
    _execution_backoff_factor: int = None
    def __init__(self, config: OperatorConfiguration) -> None:
        """
        Initialize the AbstractOperator instance with configuration.

        :param config: The operator configuration object.
        :type config: Dict[str, Any]
        """
        super().__init__()
        self._config = config
        
        self._operator_id = "OPERATOR | " + self._config.operator_operator_id
        self._operator_type = self._config.operator_operator_type
        self._enabled = self._config.operator_enabled
        self._execution_timeout = self._config.execution_timeout
        self._execution_max_retry = self._config.execution_max_retry
        self._execution_backoff_factor = self._config.execution_backoff_factor

        self._llm_component = []
        self._tool_component = []
        
        str_llm_component = ["LLM | " + llm_component for llm_component in self._config.operator_llm_component]
        list_of_initiated_llm = AbstractLanguageModel.get_llm_ids()
        for llm_component in str_llm_component:
            if llm_component in list_of_initiated_llm:
                self._llm_component.append(AbstractLanguageModel.get_llm_instance_by_id(llm_id = llm_component))
            else:
                raise ValueError(f"❌ LLM ID {llm_component} is not initiated.")
        
        str_tool_component = ["TOOL | " + tool_component for tool_component in self._config.operator_tool_component]
        list_of_initiated_tool = AbstractTool.get_tool_ids()
        for tool_component in str_tool_component:
            if tool_component in list_of_initiated_tool:
                self._tool_component.append(AbstractTool.get_tool_instance_by_id(tool_id = tool_component))
            else:
                raise ValueError(f"❌ Tool ID {tool_component} is not initiated.")
                
        if self._operator_id in self.__class__._operator_instances_by_id.keys():
            raise ValueError(f"❌ Operator ID {self._operator_id} is already initiated.")
        else:
            self.__class__._operator_instances_by_id[self._operator_id] = self
        
    @classmethod
    def get_operator_ids(cls) -> List[str]:
        """
        Get the list of operator IDs.

        :return: The list of operator IDs.
        :rtype: List[str]
        """
        return cls._operator_instances_by_id.keys()
    @classmethod
    def get_operator_instance_by_id(cls, operator_id) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._operator_instances_by_id.get(operator_id, None)

    @abstractmethod
    async def run(self, input_message: AbstractPrompt, **kwargs) -> Tuple[AbstractPrompt, Dict[uuid.UUID, List[uuid.UUID]]]:
        """
        Run the operator on the input data.
        """
        pass