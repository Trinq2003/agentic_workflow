from typing import List, Any

from base_classes.graph.node import MemoryRequiredGraphNode, SystemComponentGraphNode
from base_classes.tool import AbstractTool
from base_classes.operator import AbstractOperator
from base_classes.llm import AbstractLanguageModel
from base_classes.prompt import AbstractPrompt
from memory.operator.operator_memory import OperatorMemory

class OperatorNode(MemoryRequiredGraphNode):
    """
    This class is used to represent an operator node in the graph.
    """
    system_component: AbstractOperator
    memory: OperatorMemory
    def __init__(self, system_component: AbstractOperator, memory: OperatorMemory, **kwargs) -> None:
        super().__init__(system_component=system_component, memory=memory, **kwargs)
        
    def execute(self, input: AbstractPrompt, **kwargs):
        operator_response = self.system_component.run(input_message=input, **kwargs)
        
class LLMNode(SystemComponentGraphNode):
    """
    This class is used to represent a language model node in the graph.
    """
    system_component: AbstractLanguageModel
    def __init__(self, system_component: AbstractLanguageModel, **kwargs) -> None:
        super().__init__(system_component, **kwargs)
        
    def execute(self, input: AbstractPrompt) -> List[AbstractPrompt]:
        raw_llm_response = self.system_component.query(prompt=input, num_responses=1)
        llm_text_response = self.system_component.get_response_texts(query_responses=raw_llm_response)[0]
        
        llm_text_response_dict = [{
            'role': 'assistant',
            'content': llm_text_response
        }]
        
        response_prompt = AbstractPrompt(prompt=llm_text_response_dict)
        history = [input, response_prompt]
        
        return history