from base_classes.graph.node import AbstractGraphNode
from base_classes.tool import AbstractTool
from base_classes.operator import AbstractOperator
from base_classes.llm import AbstractLanguageModel

class InnerOperatorNode(AbstractGraphNode):
    """
    This class is used to represent an inner operator node in the graph.
    """
    _container_operator_id: str = None
    _container_operator: AbstractOperator = None
    def __init__(self, container_operator_id: str, description: str = '', **kwargs) -> None:
        super().__init__(description = description)
        self._container_operator_id = container_operator_id
        self._container_operator = AbstractOperator.get_operator_instance_by_id(operator_id = self._container_operator_id)

class OperatorNode(AbstractGraphNode):
    """
    This class is used to represent an operator node in the graph.
    """
    _operator_id: str = None
    _operator: AbstractOperator = None
    def __init__(self, operator_id: str, description: str = '') -> None:
        super().__init__(description = description)
        self._operator_id = operator_id
        self._operator = AbstractOperator.get_operator_instance_by_id(operator_id = operator_id)
        
class LLMNode(InnerOperatorNode):
    """
    This class is used to represent a language model node in the graph.
    """
    _llm_id: str = None
    _llm: AbstractLanguageModel = None
    def __init__(self, llm_id: str, container_operator_id: str, description: str = '') -> None:
        super().__init__(container_operator_id = container_operator_id, description = description)
        self._llm_id = llm_id
        self._llm = AbstractLanguageModel.get_llm_instance_by_id(llm_id = llm_id)
        
class ToolNode(InnerOperatorNode):
    """
    This class is used to represent a tool node in the graph.
    """
    _tool_id: str = None
    _tool: AbstractTool = None
    def __init__(self, tool_id: str, container_operator_id: str, description: str = '') -> None:
        super().__init__(container_operator_id = container_operator_id, description = description)
        self._tool_id = tool_id
        self._tool = AbstractTool.get_tool_instance_by_id(tool_id = tool_id)
        