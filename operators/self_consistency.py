from openai.types.chat import ChatCompletion
from typing import Union, List, Dict
import textwrap

from base_classes.operator import AbstractOperator
from configuration.operator_configuration import SelfConsistencyOperatorConfiguration
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from prompt.user_message import UserMessagePrompt
from prompt.few_shot import FewShotPrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.demonstration_sampling import DemonstrationSamplingTool
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.datatypes.data_item import PromptDataItem
from operators.op_prompt import SC_ENSEMBLE_PROMPT

class SelfConsistencyOperator(AbstractOperator):
    """
    Self-Consistency Operator for generating multiple responses and selecting the most consistent one.
    Link: https://arxiv.org/pdf/2203.11171
    """
    _config: SelfConsistencyOperatorConfiguration = None
    _sc_llm: AbstractLanguageModel = None

    def __init__(self, config: SelfConsistencyOperatorConfiguration) -> None:
        super().__init__(config=config)
        self._sc_llm = self._llm_component[0]  # Only 1 LLM component is allowed for Self-Consistency operator
        
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        """
        Run the Self-Consistency operator to generate multiple responses and select the most consistent one.

        :param input_message: The input message to process.
        :type input_message: Union[UserMessagePrompt, AssistantMessagePrompt]
        :return: The most consistent response from the generated responses.
        :rtype: AssistantMessagePrompt
        """
        self.memory_block: AbstractMemoryBlock = AbstractMemoryBlock()
        # Store the input message in memory block
        input_message_mem_atom = AbstractMemoryAtom(
            data=PromptDataItem(content=input_message, source="user")
        )
        self.memory_block.add_memory_atom(input_message_mem_atom)
        
        # Generate multiple responses using the configured LLM
        sc_reponses: List[List[Dict]] = []
        sc_mem_atoms: List[List[AbstractMemoryAtom]] = [[] for _ in range(self._config.self_consistency_num_of_iterations)]
        for iteration in range(self._config.self_consistency_num_of_iterations):
            import concurrent.futures
            
            def process_sc(reasoning_paths: List[Dict]) -> Dict:
                """
                Process a single reasoning path for self-consistency.
                """
                sc_prompt: List[Dict] = [{
                                            "role": "user",
                                            "content": textwrap.dedent(
                                                f"""
                                                Given the question described as follows: {input_message.prompt[0]["content"]}
                                                Several solutions have been generated to address the given question. They are as follows:
                                                """
                                            )
                                        }]
                sc_prompt.extend(reasoning_paths)
                sc_prompt.append({"role": "assistant", "content": "Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution."})
                
                reasoning_path = self._sc_llm.query(sc_prompt, num_responses=1)
                reasoning_path_mem_atom = AbstractMemoryAtom(
                    data=PromptDataItem(content=reasoning_path, source=self._sc_llm)
                )
                sc_mem_atoms[iteration].append(reasoning_path_mem_atom)
                if iteration == 0:
                    input_message_mem_atom.requiring_atom.append(reasoning_path_mem_atom.mem_atom_id)
                    reasoning_path_mem_atom.required_atom.append(input_message_mem_atom.mem_atom_id)
                else:
                    for mem_atom in sc_mem_atoms[iteration - 1]:
                        reasoning_path_mem_atom.required_atom.append(mem_atom.mem_atom_id)
                        mem_atom.requiring_atom.append(reasoning_path_mem_atom.mem_atom_id)
                self.memory_block.add_memory_atom(reasoning_path_mem_atom)
                return {"role": "assistant", "content": reasoning_path.choices[0].message.content}
            
            self.logger.debug(f"Start self-consistency iteration {iteration + 1} with {self._config.self_consistency_num_of_samples} reasoning paths working in parallel.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.self_consistency_num_of_samples) as executor:
                results = list(executor.map(process_sc, sc_reponses))
                sc_reponses = results
                self.logger.debug(f"Self-consistency iteration {iteration + 1} result: {sc_reponses}")