from openai.types.chat import ChatCompletion
from typing import Union, List, Dict, Any, Tuple
import uuid
import textwrap

from base_classes.operator import AbstractOperator
from configuration.operator_configuration import CoTOperatorConfiguration
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from prompt.system_message import SystemMessagePrompt
from prompt.user_message import UserMessagePrompt
from prompt.few_shot import FewShotPrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.demonstration_sampling import DemonstrationSamplingTool
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.datatypes.data_item import PromptDataItem
from operators.op_prompt import COT_INSTRUCTION_PROMPT

class CoTOperator(AbstractOperator):
    """
    Inspired by Auto-Cot: Automatic Chain-of-Thought Prompting paper by AWS.
    Link: https://arxiv.org/pdf/2210.03493
    """
    _config: CoTOperatorConfiguration = None
    _construct_cot_tool: DemonstrationSamplingTool = None
    _cot_llm: AbstractLanguageModel = None
    def __init__(self, config: CoTOperatorConfiguration) -> None:
        super().__init__(config = config)
        self._construct_cot_tool = self._tool_component[0] # Only 1 tool component is allowed for CoT operator. This tool is used to construct the CoT prompt.
        self._cot_llm = self._llm_component[0] # Only 1 LLM component is allowed for CoT operator
        
    async def _demonstration_sampling(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> FewShotPrompt:
        """
        This method is used to demonstrate the sampling of the CoT operator.
        """

        demonstration_samples: List[Dict[str, Any]] = await self._construct_cot_tool.execute(input_message = input_message)
        few_shot_demonstration_samples = FewShotPrompt(
            prompt = [
                {
                    "role": "tool",
                    "content": sample
                } for sample in demonstration_samples
            ]
        )
        return few_shot_demonstration_samples
        
    async def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> Tuple[AbstractMemoryAtom, Dict[uuid.UUID, List[uuid.UUID]]]:
        """
        This method is used to run the CoT operator.
        """
        # Operator execution logic
        instruction_prompt = SystemMessagePrompt(prompt = [{"role": "system", "content": COT_INSTRUCTION_PROMPT}])
        demonstration_samples: FewShotPrompt = await self._demonstration_sampling(input_message = input_message)
        user_prompt = UserMessagePrompt(prompt = [{"role": "user", "content": f"The user query is: \n <problem>{input_message.prompt[0].get('content')}</problem>. \n Please generate the plan following the given instruction."}])

        final_cot_prompt = UserMessagePrompt([{
            "role": "user",
            "content": instruction_prompt.prompt[0]["content"] + demonstration_samples.prompt[0]["content"] + user_prompt.prompt[0]["content"]
        }])
        self.logger.debug(f"Final CoT prompt: {final_cot_prompt.prompt}")
        cot_query_answer: ChatCompletion = self._cot_llm.query(query = final_cot_prompt, num_responses= 1)
        self.logger.debug(f"CoT query answer: {self._cot_llm.get_response_texts(cot_query_answer)}")
        cot_answer: AssistantMessagePrompt = AssistantMessagePrompt(
            prompt = [
                {
                    "role": "assistant",
                    "content": cot_query_answer.choices[0].message.content
                }
            ]
        )

        self.logger.debug(f"CoT answer: {cot_answer.prompt}")

        demonstration_samples_memory_atom = AbstractMemoryAtom(data=PromptDataItem(demonstration_samples))
        cot_prompt_memory_atom = AbstractMemoryAtom(data=PromptDataItem(final_cot_prompt))
        cot_answer_memory_atom = AbstractMemoryAtom(data=PromptDataItem(cot_answer))
        dependency_graph = {
            demonstration_samples_memory_atom.mem_atom_id: [cot_prompt_memory_atom.mem_atom_id],
            cot_prompt_memory_atom.mem_atom_id: [cot_answer_memory_atom.mem_atom_id],
            cot_answer_memory_atom.mem_atom_id: []
        }

        self.logger.debug(f"Dependency graph: {dependency_graph}")

        return cot_answer_memory_atom, dependency_graph
    
