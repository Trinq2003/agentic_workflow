from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Union, List, Dict, Tuple
import uuid
from copy import deepcopy

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
from memory.utils import visualize_dependency_graph

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
    
    def _construct_debate_message(self, agent_contexts, question, idx) -> List[Dict]:
        constructed_message = "These are the solutions to the problem from other agents:\n"
        for agent_id, agent in enumerate(agent_contexts):
            agent_response = agent[idx]["content"]
            response_parsed = f"Solution and reasoning from agent {agent_id + 1}:\n{agent_response}\n"

            constructed_message += response_parsed

        constructed_message += f"\n\nUsing the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that of other agents step by step. Original question: {question}"
        constructed_message_obj = {"role": "user", "content": constructed_message}
        # self.logger.debug(f"Constructed message: {constructed_message_obj}")
        return constructed_message_obj
    
    async def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> Tuple[AbstractPrompt, Dict[uuid.UUID, List[uuid.UUID]]]:
        # 1. Create input message memory atom
        input_prompt_memory_atom = AbstractMemoryAtom(data=PromptDataItem(deepcopy(input_message)))
        self.logger.debug(f"[MEMORY] Created input message memory atom: {input_prompt_memory_atom.mem_atom_id}")
        
        # Extract the content from the input message
        input_content = input_message.prompt[0]["content"] if input_message.prompt else ""
        
        # Initialize debate contexts and memory tracking
        debate_contexts: List[List[Dict]] = [[{"role": "user", "content": input_content}] for _ in range(self._num_of_debaters)]
        
        # Track memory atoms for each debater across rounds
        # Structure: debater_idx -> round_idx -> [query_memory_atom, response_memory_atom, debate_message_memory_atom]
        debater_memory_atoms: List[List[List[AbstractMemoryAtom]]] = [[] for _ in range(self._num_of_debaters)]
        
        self.logger.debug(f"[INIT] Starting debate with {self._num_of_debaters} debaters for {self._num_of_rounds} rounds")
        
        for round_idx in range(self._num_of_rounds):
            self.logger.debug(f"[ROUND] Starting round {round_idx + 1}/{self._num_of_rounds}")
            
            # For each debater in this round
            for debater_idx in range(self._num_of_debaters):
                self.logger.debug(f"[DEBATER] Processing debater {debater_idx + 1} in round {round_idx + 1}")
                
                # Initialize memory atoms for this debater in this round
                debater_memory_atoms[debater_idx].append([])
                
                # Round 0: Use input message directly
                if round_idx == 0:
                    self.logger.debug(f"[ROUND 0] Debater {debater_idx + 1} using input message directly")
                    query_prompt = AbstractPrompt(prompt=debate_contexts[debater_idx])
                    query_memory_atom = AbstractMemoryAtom(data=PromptDataItem(deepcopy(query_prompt)))
                    debater_memory_atoms[debater_idx][round_idx].append(query_memory_atom)
                    self.logger.debug(f"[MEMORY] Created query memory atom for debater {debater_idx + 1} round {round_idx + 1}: {query_memory_atom.mem_atom_id}")
                
                # Round > 0: Create debate message from other debaters' responses
                else:
                    self.logger.debug(f"[ROUND {round_idx + 1}] Creating debate message for debater {debater_idx + 1}")
                    
                    # Get other debaters' contexts (excluding current debater)
                    other_debaters_contexts = debate_contexts[:debater_idx] + debate_contexts[debater_idx + 1:]
                    
                    # Create debate message
                    debate_message = self._construct_debate_message(other_debaters_contexts, input_content, 2 * round_idx - 1)
                    debate_contexts[debater_idx].append(debate_message)
                    
                    # Create memory atom for the debate message
                    debate_message_prompt = AbstractPrompt(prompt=[debate_message])
                    debate_message_memory_atom = AbstractMemoryAtom(data=PromptDataItem(deepcopy(debate_message_prompt)))
                    debater_memory_atoms[debater_idx][round_idx].append(debate_message_memory_atom)
                    self.logger.debug(f"[MEMORY] Created debate message memory atom for debater {debater_idx + 1} round {round_idx + 1}: {debate_message_memory_atom.mem_atom_id}")
                    
                    # Create query prompt including the debate message
                    query_prompt = AbstractPrompt(prompt=debate_contexts[debater_idx])
                    query_memory_atom = AbstractMemoryAtom(data=PromptDataItem(deepcopy(query_prompt)))
                    debater_memory_atoms[debater_idx][round_idx].append(query_memory_atom)
                    self.logger.debug(f"[MEMORY] Created query memory atom for debater {debater_idx + 1} round {round_idx + 1}: {query_memory_atom.mem_atom_id}")
                
                # Get response from LLM
                self.logger.debug(f"[LLM] Querying LLM for debater {debater_idx + 1} in round {round_idx + 1}")
                completion = self._list_of_debaters[debater_idx].query(query_prompt)
                
                # Create response message and memory atom
                response_content = completion.choices[0].message.content
                response_message = {
                    "role": "assistant",
                    "content": response_content
                }
                debate_contexts[debater_idx].append(response_message)
                
                assistant_message = AssistantMessagePrompt(prompt=[response_message])
                response_memory_atom = AbstractMemoryAtom(data=PromptDataItem(deepcopy(assistant_message)))
                debater_memory_atoms[debater_idx][round_idx].append(response_memory_atom)
                self.logger.debug(f"[MEMORY] Created response memory atom for debater {debater_idx + 1} round {round_idx + 1}: {response_memory_atom.mem_atom_id}")
                self.logger.debug(f"[RESPONSE] Debater {debater_idx + 1} response length: {len(response_content)} characters")
        
        # Create final response from all debaters' last responses
        self.logger.debug(f"[FINAL] Creating final combined response")
        final_responses = []
        for debater_idx, context in enumerate(debate_contexts):
            if context and len(context) > 0:
                final_responses.append(context[-1]["content"])
        
        # Combine all final responses
        combined_response = "\n" + "="*50 + "\n".join([f"Debater {i+1}: {response}" for i, response in enumerate(final_responses)])
        
        final_assistant_message = AssistantMessagePrompt(prompt=[{
            "role": "assistant", 
            "content": combined_response
        }])
        final_assistant_message_memory_atom = AbstractMemoryAtom(data=PromptDataItem(deepcopy(final_assistant_message)))
        self.logger.debug(f"[MEMORY] Created final response memory atom: {final_assistant_message_memory_atom.mem_atom_id}")
        
        # Build dependency graph
        self.logger.debug(f"[DEPENDENCY] Building dependency graph")
        dependency_graph = {}
        
        # First, initialize all memory atoms in the dependency graph
        dependency_graph[input_prompt_memory_atom.mem_atom_id] = []
        
        # Initialize all debater memory atoms
        for debater_idx in range(self._num_of_debaters):
            for round_idx in range(self._num_of_rounds):
                round_memory_atoms = debater_memory_atoms[debater_idx][round_idx]
                for memory_atom in round_memory_atoms:
                    dependency_graph[memory_atom.mem_atom_id] = []
        
        # Initialize final response memory atom
        dependency_graph[final_assistant_message_memory_atom.mem_atom_id] = []
        
        # Now build the dependencies
        for debater_idx in range(self._num_of_debaters):
            for round_idx in range(self._num_of_rounds):
                round_memory_atoms = debater_memory_atoms[debater_idx][round_idx]
                
                if round_idx == 0:
                    # Round 0: Query depends on input message
                    query_memory_atom = round_memory_atoms[0]  # Query memory atom
                    response_memory_atom = round_memory_atoms[1]  # Response memory atom
                    
                    # Input message is depended on by query
                    dependency_graph[input_prompt_memory_atom.mem_atom_id].append(query_memory_atom.mem_atom_id)
                    self.logger.debug(f"[DEPENDENCY] Input {input_prompt_memory_atom.mem_atom_id} is depended on by query {query_memory_atom.mem_atom_id}")
                    
                    # Query is depended on by response
                    dependency_graph[query_memory_atom.mem_atom_id].append(response_memory_atom.mem_atom_id)
                    self.logger.debug(f"[DEPENDENCY] Query {query_memory_atom.mem_atom_id} is depended on by response {response_memory_atom.mem_atom_id}")
                
                else:
                    # Round > 0: Debate message depends on previous round responses from other debaters
                    debate_message_memory_atom = round_memory_atoms[0]  # Debate message memory atom
                    query_memory_atom = round_memory_atoms[1]  # Query memory atom
                    response_memory_atom = round_memory_atoms[2]  # Response memory atom
                    
                    # Get previous round response memory atoms from other debaters
                    for other_debater_idx in range(self._num_of_debaters):
                        if other_debater_idx != debater_idx:
                            prev_response_memory_atom = debater_memory_atoms[other_debater_idx][round_idx - 1][-1]  # Last memory atom is response
                            # Previous response is depended on by debate message
                            dependency_graph[prev_response_memory_atom.mem_atom_id].append(debate_message_memory_atom.mem_atom_id)
                            self.logger.debug(f"[DEPENDENCY] Previous response {prev_response_memory_atom.mem_atom_id} is depended on by debate message {debate_message_memory_atom.mem_atom_id}")
                    
                    # Debate message is depended on by query
                    dependency_graph[debate_message_memory_atom.mem_atom_id].append(query_memory_atom.mem_atom_id)
                    self.logger.debug(f"[DEPENDENCY] Debate message {debate_message_memory_atom.mem_atom_id} is depended on by query {query_memory_atom.mem_atom_id}")
                    
                    # Query is depended on by response
                    dependency_graph[query_memory_atom.mem_atom_id].append(response_memory_atom.mem_atom_id)
                    self.logger.debug(f"[DEPENDENCY] Query {query_memory_atom.mem_atom_id} is depended on by response {response_memory_atom.mem_atom_id}")
        
        # Final response depends on all last round responses
        last_round_responses = []
        for debater_idx in range(self._num_of_debaters):
            last_response_memory_atom = debater_memory_atoms[debater_idx][-1][-1]  # Last memory atom of last round
            last_round_responses.append(last_response_memory_atom.mem_atom_id)
            # Last round response is depended on by final response
            dependency_graph[last_response_memory_atom.mem_atom_id].append(final_assistant_message_memory_atom.mem_atom_id)
            self.logger.debug(f"[DEPENDENCY] Last round response {last_response_memory_atom.mem_atom_id} is depended on by final response {final_assistant_message_memory_atom.mem_atom_id}")
        
        self.logger.debug(f"[DEPENDENCY] Final response {final_assistant_message_memory_atom.mem_atom_id} depends on last round responses: {last_round_responses}")
        
        # Log memory atom summary
        self.logger.debug(f"[SUMMARY] Memory atom creation summary:")
        self.logger.debug(f"  - Input message: {input_prompt_memory_atom.mem_atom_id}")
        for debater_idx in range(self._num_of_debaters):
            self.logger.debug(f"  - Debater {debater_idx + 1}:")
            for round_idx in range(self._num_of_rounds):
                round_atoms = debater_memory_atoms[debater_idx][round_idx]
                atom_ids = [atom.mem_atom_id for atom in round_atoms]
                self.logger.debug(f"    Round {round_idx + 1}: {atom_ids}")
        self.logger.debug(f"  - Final response: {final_assistant_message_memory_atom.mem_atom_id}")
        
        self.logger.debug(f"[DEPENDENCY] Final dependency graph: {visualize_dependency_graph(dependency_graph)}")
        
        return final_assistant_message, dependency_graph
        
