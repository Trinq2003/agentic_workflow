from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Union

from base_classes.operator import AbstractOperator
from configuration.operator_configuration import CoTOperatorConfiguration
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from prompt.user_message import UserMessagePrompt
from prompt.few_shot import FewShotPrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.demonstration_sampling import DemonstrationSamplingTool
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.datatypes.data_item import PromptDataItem

class CoTOperator(AbstractOperator):
    """
    Inspired by Auto-Cot: Automatic Chain-of-Thought Prompting paper by AWS.
    Link: https://arxiv.org/pdf/2210.03493

    Args:
        AbstractOperator (_type_): _description_
    """
    _config: CoTOperatorConfiguration = None
    _construct_cot_tool: DemonstrationSamplingTool = None
    _cot_llm: AbstractLanguageModel = None
    def __init__(self, config: CoTOperatorConfiguration) -> None:
        super().__init__(config = config)
        self._construct_cot_tool = self._tool_component[0] # Only 1 tool component is allowed for CoT operator. This tool is used to construct the CoT prompt.
        self._cot_llm = self._llm_component[0] # Only 1 LLM component is allowed for CoT operator
        
    def _demonstration_sampling(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> FewShotPrompt:
        """
        This method is used to demonstrate the sampling of the CoT operator.
        """
        return FewShotPrompt(self._construct_cot_tool.execute(input_message = input_message))
    
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        """
        This method is used to run the CoT operator.
        """
        history_memory_block: AbstractMemoryBlock = AbstractMemoryBlock()
        history_memory_block.add_memory_atom(AbstractMemoryAtom(data = PromptDataItem(content = input_message, source = "user")))
        input_message.prompt[0]["content"] = input_message.prompt[0]["content"] + "\nLet's think step by step.\n"
        
        demonstration_samples: FewShotPrompt = self._demonstration_sampling(input_message = input_message)
        history_memory_block.add_memory_atom(AbstractMemoryAtom(data = PromptDataItem(content = demonstration_samples, source = self._construct_cot_tool)))
        
        final_cot_prompt = demonstration_samples.prompt.append(input_message.prompt[0])
        
        cot_query_answer: ChatCompletion = self._cot_llm.query(prompt = final_cot_prompt, num_responses= 1)
        cot_answer: AssistantMessagePrompt = AssistantMessagePrompt(cot_query_answer.choices[0].message.content)
        history_memory_block.add_memory_atom(AbstractMemoryAtom(data = PromptDataItem(content = cot_answer, source = self._cot_llm)))
        
        self.memory_block = history_memory_block
        
        return AssistantMessagePrompt(cot_answer)
    
