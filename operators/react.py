from typing import Union, Dict, Any, List
from collections.abc import Callable
from annotated_docs.json_schema import as_json_schema
import json

from base_classes.operator import AbstractOperator
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.datatypes.data_item import PromptDataItem
from configuration.operator_configuration import ReActOperatorConfiguration
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt
from prompt.tool_message import ToolMessagePrompt
from tools.tool_chooser import ToolChooserTool

class ReactOperator(AbstractOperator):
    """
    This operator is used to react to the user message.
    Operator design is inspired by ReAct: Synergizing Reasoning and Acting in Language Models paper by Shuyun Yao et al.
    Link: https://arxiv.org/abs/2210.03629.pdf
    """
    _config: ReActOperatorConfiguration = None
    _tool_chooser: ToolChooserTool = None
    _callable_tools: Dict[str, Dict[str, Any]] = {}
    _max_iterations: int = 10
    _reasoning_llm: AbstractLanguageModel = None
    def __init__(self, config: ReActOperatorConfiguration) -> None:
        super().__init__(config = config)
        self._max_iterations = config.max_iterations
        self._callable_tools: Dict[str, AbstractTool] = {}
        self._tool_chooser = ToolChooserTool.get_tool_instance_by_id(tool_id = "TOOL | " + config.tool_tool_choser)
        callable_tools = config.tool_callable
        for callable_tool in callable_tools:
            tool = AbstractTool.get_tool_instance_by_id(tool_id = "TOOL | " + callable_tool)
            self._callable_tools[callable_tool] = {'tool': tool}
            self._callable_tools[callable_tool]['description'] = as_json_schema(tool.__class__)
        
        self._reasoning_llm = self._llm_component[0] # Only 1 LLM component is allowed for React operator.
        
    def _choose_tool_id(self, input_message: str) -> Dict:
        """
        This method is used to choose the tool for the React operator. This method returns natural text for the tool to be used, in form of a prompt (tool call prompt).
        :return: The tool to be used for the React operator (natural text).
        """
        tools_to_call: List[Dict] = self._tool_chooser.execute(input_message = input_message)
        return {"role": "tool",
            "content": f"You can call the following functions: {tools_to_call}",
            "tool_call_id": "tool_chooser"
            }
    
    def _get_observation_by_executing_tool(self, list_of_tools: List) -> str:
        _observation = ""
        for tool in list_of_tools:
            tool_id = "TOOL | " + tool['id']
            tool_params = json.loads(tool['function']['arguments'])
            tool_instance: AbstractTool = self._callable_tools[tool_id]['tool']
            result = tool_instance.execute(function_params = tool_params)
            self.memory_block.add_memory_atom(AbstractMemoryAtom(data = PromptDataItem(content = ToolMessagePrompt(prompt = {'role': 'tool', 'content': str(result), 'tool_call_id': tool_id}), source = tool_instance)))
            _observation += f"Tool: {tool_id}\n"
            _observation += f"Result: {result}\n"
            
        return f"<observation>{_observation}</observation>"
    
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        """
        This method is used to run the React operator.
        """
        self.memory_block: AbstractMemoryBlock = AbstractMemoryBlock()
        REACT_MESSAGE = [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer multistep questions by sequentially calling functions. Follow a pattern of THOUGHT (reason step-by-step about which function to call next in <though></though> XML tags), ACTION (call a function to as a next step towards the final answer in <action></action> XML tags), OBSERVATION (output of the function in <observation></observation> XML tags). Reason step by step which actions to take to get to the answer. When you get the result, encloses it inside <finish></finish> XML tag. Only call functions with arguments coming verbatim from the user or the output of other functions."
            },
            {
                "role": "user",
                "content": "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.</thought>",
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"wikipedia\": {{\"entity\": {{\"type\": \"string\", \"description\": \"The entity to search for in Wikipedia.\"}}}}}}]",
                "tool_call_id": "tool_chooser"
            },
            {
                "role": "assistant",
                "content": "<action>Action 1: wikipedia[Colorado orogeny]</action>",
                "tool_calls": [
                    {
                        "id": "wikipedia",
                        "function": {
                            "arguments": "{{\"entity\": \"Colorado orogeny\"}}",
                            "name": "wikipedia"
                        },
                        "type": "function",
                    }
                ]
            },
            {
                "role": "tool",
                "content": "<observation>Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.</observation>",
                "tool_call_id": "environment"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.</thought>"
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"look_up\": {{\"keyword\": {{\"type\": \"string\", \"description\": \"Lookup the keywords in the given text.\"}}}}}}]",
                "tool_call_id": "tool_chooser"
            },
            {
                "role": "assistant",
                "content": "<action>Action 2: look_up[eastern sector]</action>",
                "tool_calls": [
                    {
                        "id": "look_up",
                        "function": {
                            "arguments": "{{\"keyword\": \"eastern sector\"}}",
                            "name": "look_up"
                        },
                        "type": "function",
                    }
                ]
            },
            {
                "role": "tool",
                "content": "<observation>Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.</observation>",
                "tool_call_id": "environment"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.</thought>"
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"look_up\": {{\"keyword\": {{\"type\": \"string\", \"description\": \"Lookup the keywords in the given text.\"}}}}}}]",
                "tool_call_id": "tool_chooser"
            },
            {
                "role": "assistant",
                "content": "<action>Action 3: look_up[high Plains]</action>",
                "tool_calls": [
                    {
                        "id": "look_up",
                        "function": {
                            "arguments": "{{\"keyword\": \"high Plains\"}}",
                            "name": "look_up"
                        },
                        "type": "function",
                    }
                ]
            },
            {
                "role": "tool",
                "content": "<observation>Observation 3: High Plains refers to one of two distinct land regions:</observation>",
                "tool_call_id": "environment"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 4: I need to instead search High Plains (United States).</thought>"
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"wikipedia\": {{\"entity\": {{\"type\": \"string\", \"description\": \"The entity to search for in Wikipedia.\"}}}}}}]",
                "tool_call_id": "tool_chooser"
            },
            {
                "role": "assistant",
                "content": "<action>Action 4: wikipedia[High Plains (United States)]</action>",
                "tool_calls": [
                    {
                        "id": "wikipedia",
                        "function": {
                            "arguments": "{{\"entity\": \"High Plains (United States)\"}}",
                            "name": "wikipedia"
                        },
                        "type": "function",
                    }
                ]
            },
            {
                "role": "tool",
                "content": "<observation>Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).</observation>",
                "tool_call_id": "environment"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.</thought><finish>1,800 to 7,000 ft</finish>"
            },
            {
                "role": "user",
                "content": input_message.prompt[0]["content"],
            },
        ]
        
        # Store the input message in memory block
        input_message_mem_atom = AbstractMemoryAtom(
            data = PromptDataItem(content = input_message, source = "user")
        )
        self.memory_block.add_memory_atom(input_message_mem_atom)
        
        for i in range(self._max_iterations):
            thought_response = self._reasoning_llm.query(prompt = REACT_MESSAGE, num_responses = 1, stop = [f"<action>"])
            thought_response_str = self._reasoning_llm.get_response_texts(query_responses = thought_response)[0]
            # Thoughts are generated as a thinking step. Consider these as a memory atom and add these thoughts to the memory block.
            thought_response_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(
                    content = AssistantMessagePrompt(
                        prompt = {'role': 'assistant', 'content': thought_response_str}
                        ), source = self._reasoning_llm
                    )
                )
            self.memory_block.add_memory_atom(thought_response_mem_atom)
            if 'finish' in thought_response_str.lower():
                break
            
            tool_chooser_response: Dict = self._choose_tool_id(input_message = thought_response_str)
            # Tool chooser was called. Consider this as a memory atom and add this to the memory block.
            tool_chooser_response_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(content = ToolMessagePrompt(prompt = tool_chooser_response), source = self._tool_chooser)
                )
            self.memory_block.add_memory_atom(tool_chooser_response_mem_atom)
            
            REACT_MESSAGE.append(tool_chooser_response)
            action = self._reasoning_llm.query(prompt = REACT_MESSAGE, num_responses = 1, stop = [f"<observation>"])
            # Reasoning LLM is called to get the tool to be used (the list of tools is given by tool chooser). Consider this as a memory atom and add this to the memory block.
            action_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(content = AssistantMessagePrompt(prompt = action), source = self._reasoning_llm)
            )
            self.memory_block.add_memory_atom(action_mem_atom)
            tool_calls_response = action['tool_calls']
            tool_observations: str = self._get_observation_by_executing_tool(input_message = tool_calls_response)
            tool_observations_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(content = ToolMessagePrompt(prompt = {'role': 'tool', 'content': tool_observations, 'tool_call_id': 'environment'}), source = self._callable_tools['TOOL | environment']['tool'])
            )
            self.memory_block.add_memory_atom(tool_observations_mem_atom)
            REACT_MESSAGE.append({'role': 'tool', 'content': tool_observations, 'tool_call_id': 'environment'})

            input_message_mem_atom.requiring_atom.append(thought_response_mem_atom.mem_atom_id)
            thought_response_mem_atom.requiring_atom.append(tool_chooser_response_mem_atom.mem_atom_id)
            tool_chooser_response_mem_atom.requiring_atom.append(action_mem_atom.mem_atom_id)
            action_mem_atom.requiring_atom.append(tool_observations_mem_atom.mem_atom_id)
            
            thought_response_mem_atom.required_atom.append(input_message_mem_atom.mem_atom_id)
            tool_chooser_response_mem_atom.required_atom.append(thought_response_mem_atom.mem_atom_id)
            action_mem_atom.required_atom.append(tool_chooser_response_mem_atom.mem_atom_id)
            tool_observations_mem_atom.required_atom.append(action_mem_atom.mem_atom_id)
        return AssistantMessagePrompt(prompt={'role': 'assistant', 'content': thought_response_str})