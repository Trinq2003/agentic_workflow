from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Union, List, Dict

from base_classes.operator import AbstractOperator
from configuration.operator_configuration import DebateOperatorConfiguration
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from prompt.user_message import UserMessagePrompt
from prompt.few_shot import FewShotPrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.demonstration_sampling import DemonstrationSamplingTool
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.datatypes.data_item import PromptDataItem

class DebateOperator(AbstractOperator):
    """
    Inspired by Improving Factuality and Reasoning in Language Models through Multiagent Debate paper by MIT and Google Brain.
    Link: https://arxiv.org/pdf/2305.14325
    """
    _config: DebateOperatorConfiguration = None
    _num_of_rounds: int = None
    _num_of_debaters: int = None
    def __init__(self, config: DebateOperatorConfiguration) -> None:
        super().__init__(config = config)
        self._num_of_rounds = self._config.debate_num_of_round
        self._num_of_debaters = self._config.debate_num_of_debaters
        self.logger.debug(f"Datebase Operator ID: {self._operator_id} | Number of rounds: {self._num_of_rounds} | Number of debaters: {self._num_of_debaters}")
        self._list_of_debaters: List[AbstractLanguageModel] = []
        for i in range(self._num_of_debaters):
            self._list_of_debaters.append(
                AbstractLanguageModel(self._config.debate_config.get(f"debater_{i + 1}", {})['llm_id'])
            )
    
    def _construct_debate_message(self, agent_contexts, question, idx):
        constructed_message = "These are the solutions to the problem from other agents:\n"
        for agent_id, agent in enumerate(agent_contexts):
            agent_response = agent[idx]["content"]
            response_parsed = f"Solution and reasoning from agent {agent_id + 1}:\n{agent_response}\n"

            constructed_message += response_parsed

        constructed_message += """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step.""".format(question)
        constructed_message_obj =  {"role": "user", "content": constructed_message}
        self.logger.debug(f"Constructed message: {constructed_message_obj}")
        return constructed_message_obj
    
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        self.memory_block: AbstractMemoryBlock = AbstractMemoryBlock()
        
        # Store the input message in memory block
        input_message_mem_atom = AbstractMemoryAtom(
            data=PromptDataItem(content=input_message, source="user")
        )
        self.memory_block.add_memory_atom(input_message_mem_atom)
        
        debate_contexts: List[List[Dict]] = [[{"role": "user", "content": input_message.text}] for debater in range(self._num_of_debaters)]
        debate_mem_atoms: List[List[AbstractMemoryAtom]] = [[] for _ in range(self._num_of_rounds)]
        self.logger.debug(f"Debate contexts: {debate_contexts}")
        for round in range(self._num_of_rounds):
            import concurrent.futures

            def process_debater(i_debate: tuple[int, List[Dict]]) -> List[Dict]:
                i, debate_context = i_debate
                if round != 0:
                    debate_contexts_other = debate_contexts[:i] + debate_contexts[i+1:] # Get context from other debaters, excluding the current one
                    debate_message = self._construct_debate_message(debate_contexts_other, input_message.text, 2 * round - 1)
                    debate_context.append(debate_message)
                completion = self._list_of_debaters[i].query(debate_context)
                assistant_message = AssistantMessagePrompt(completion.choices[0].message.content)
                self.logger.debug(f"Debater {i + 1} response: {assistant_message.text}")
                debate_context.append(assistant_message)
                
                # Store the assistant message in memory block
                assistant_message_mem_atom = AbstractMemoryAtom(
                    data=PromptDataItem(content=assistant_message, source=self._list_of_debaters[i])
                )
                self.memory_block.add_memory_atom(assistant_message_mem_atom)
                debate_mem_atoms[round].append(assistant_message_mem_atom)
                if round == 0:
                    input_message_mem_atom.requiring_atom.append(assistant_message_mem_atom.mem_atom_id)
                    assistant_message_mem_atom.required_atom.append(input_message_mem_atom.mem_atom_id)
                else:
                    for idx, mem_atom in enumerate(debate_mem_atoms[round - 1]):
                        if idx == i:
                            continue
                        assistant_message_mem_atom.required_atom.append(mem_atom.mem_atom_id)
                        mem_atom.requiring_atom.append(assistant_message_mem_atom.mem_atom_id)
                
                return debate_context
            
            self.logger.debug(f"Start debate round {round + 1} with {len(debate_contexts)} debaters working in parallel.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(debate_contexts)) as executor:
                results = list(executor.map(process_debater, enumerate(debate_contexts)))
                debate_contexts = results
                self.logger.debug(f"Debate round {round + 1} result: {debate_contexts}")
        
