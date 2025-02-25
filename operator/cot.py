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
        demonstration_samples: FewShotPrompt = self._demonstration_sampling(input_message = input_message)
        final_cot_prompt = demonstration_samples.prompt.append(input_message.prompt)
        
        cot_query_answer: ChatCompletion = self._cot_llm.query(prompt = final_cot_prompt, num_responses= 1)
        cot_answer: ChatCompletionMessageParam = cot_query_answer.choices[0].message
        
        return AssistantMessagePrompt(cot_answer)
    
