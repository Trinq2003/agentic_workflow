from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Union, List, Dict, Tuple
import uuid

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
from base_classes.prompt import AbstractPrompt

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
        self._num_of_rounds = self._config.debate_num_of_rounds
        self._num_of_debaters = self._config.debate_num_of_debaters
        self.logger.debug(f"Debate Operator ID: {self._operator_id}; Number of rounds: {self._num_of_rounds}; Number of debaters: {self._num_of_debaters}")
        self._list_of_debaters: List[AbstractLanguageModel] = []
        
        debater_configs = self._config.debate_config_individual_configs
        self.logger.debug(f"Debater configs: {debater_configs}")

        for i in range(self._num_of_debaters):
            if i < len(debater_configs) and 'llm_id' in debater_configs[i]:
                llm_id = "LLM | " + debater_configs[i]['llm_id']
                # Get the LLM instance by ID from the LLM component
                llm_instance = AbstractLanguageModel.get_llm_instance_by_id(llm_id)
                if llm_instance:
                    self._list_of_debaters.append(llm_instance)
                    self.logger.debug(f"Added debater {debater_configs[i]['debater']} with LLM ID: {llm_id}")
                else:
                    self.logger.error(f"LLM instance with ID '{llm_id}' not found for debater {i+1}")
                    raise ValueError(f"LLM instance with ID '{llm_id}' not found for debater {i+1}")
            else:
                self.logger.error(f"Missing or invalid configuration for debater {i+1}")
                raise ValueError(f"Missing or invalid configuration for debater {i+1}")
        
        self.logger.debug(f"Successfully initialized {len(self._list_of_debaters)} debaters")
    
    def _construct_debate_message(self, agent_contexts, question, idx):
        constructed_message = "These are the solutions to the problem from other agents:\n"
        for agent_id, agent in enumerate(agent_contexts):
            agent_response = agent[idx]["content"]
            response_parsed = f"Solution and reasoning from agent {agent_id + 1}:\n{agent_response}\n"

            constructed_message += response_parsed

        constructed_message += f"\n\nUsing the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that of other agents step by step. Original question: {question}"
        constructed_message_obj = {"role": "user", "content": constructed_message}
        self.logger.debug(f"Constructed message: {constructed_message_obj}")
        return constructed_message_obj
    
    async def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> Tuple[AbstractPrompt, Dict[uuid.UUID, List[uuid.UUID]]]:
        # Extract the content from the input message
        input_content = input_message.prompt[0]["content"] if input_message.prompt else ""
        
        debate_contexts: List[List[Dict]] = [[{"role": "user", "content": input_content}] for debater in range(self._num_of_debaters)]
        self.logger.debug(f"Initial debate contexts: {debate_contexts}")
        for round in range(self._num_of_rounds):
            import concurrent.futures

            def process_debater(i_debate: tuple[int, List[Dict]]) -> List[Dict]:
                i, debate_context = i_debate
                if round != 0:
                    debate_contexts_other = debate_contexts[:i] + debate_contexts[i+1:] # Get context from other debaters, excluding the current one
                    debate_message = self._construct_debate_message(debate_contexts_other, input_message.text, 2 * round - 1)
                    debate_context.append(debate_message)
                
                self.logger.debug(f"Debate context for debater {i + 1}: {debate_context}")

                query_prompt = AbstractPrompt(prompt=debate_context)
                completion = self._list_of_debaters[i].query(query_prompt)
                
                assistant_message = AssistantMessagePrompt(prompt=[{
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                }])
                self.logger.debug(f"Debater {i + 1} response: {assistant_message.prompt[0]['content']}")
                debate_context.append({
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                })
                
                return debate_context
            
            self.logger.debug(f"Start debate round {round + 1} with {len(debate_contexts)} debaters working in parallel.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(debate_contexts)) as executor:
                results = list(executor.map(process_debater, enumerate(debate_contexts)))
                debate_contexts = results
                self.logger.debug(f"Debate round {round + 1} result: {debate_contexts}")
        
        # Create final response from the last round
        final_responses = []
        for i, context in enumerate(debate_contexts):
            if context and len(context) > 0:
                final_responses.append(context[-1]["content"])
        
        # Combine all final responses
        combined_response = "\n\n".join([f"Debater {i+1}: {response}" for i, response in enumerate(final_responses)])
        
        final_assistant_message = AssistantMessagePrompt(prompt=[{
            "role": "assistant", 
            "content": combined_response
        }])
        
        # Create dependency graph
        dependency_graph = {}
        
        return final_assistant_message, dependency_graph
        
