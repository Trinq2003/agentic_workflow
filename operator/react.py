from typing import Union, Dict, Any, List
from collections.abc import Callable
from annotated_docs.json_schema import as_json_schema
import json

from base_classes.operator import AbstractOperator
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from configuration.operator_configuration import ReActOperatorConfiguration
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.tool_chooser import ToolChooserTool

class ReactOperator(AbstractOperator):
    """
    This operator is used to react to the user message.
    """
    _config: ReActOperatorConfiguration = None
    _tool_chooser: ToolChooserTool = None
    _callable_tools: Dict[str, Dict[str, Any]] = {}
    _max_iterations: int = 10
    _reasoning_llm: AbstractLanguageModel = None
    def __init__(self, config: ReActOperatorConfiguration) -> None:
        super().__init__(config = config)
        self._max_iterations = config.max_iterations
        self._callable_tools: Dict[str, Any] = {}
        self._tool_chooser = ToolChooserTool.get_tool_instance_by_id(tool_id = "TOOL | " + config.tool_tool_choser)
        callable_tools = config.tool_callable
        for callable_tool in callable_tools:
            tool = AbstractTool.get_tool_instance_by_id(tool_id = "TOOL | " + callable_tool)
            self._callable_tools[callable_tool] = {'tool': tool}
            self._callable_tools[callable_tool]['description'] = as_json_schema(tool.__class__)
        
        self._reasoning_llm = self._llm_component[0] # Only 1 LLM component is allowed for React operator.
        
    def _choose_tool_id(self, input_message: str) -> List[str]:
        """
        This method is used to choose the tool for the React operator.
        :return: The tool ID(s) for the React operator.
        """
        return self._tool_chooser.execute(input_message = input_message)
    
    def _get_observation_by_executing_tool(self, input_message: str) -> str:
        tool_ids: List[str] = self._choose_tool_id(input_message = input_message)
        
        function_params = json.loads(input_message) # TODO: This is a placeholder. Replace this by parsing the input_message.
        
        results = {}
        observation_str = "<observation>"
        for tool_id in tool_ids:
            tool: AbstractTool = self._callable_tools[tool_id]['tool']
            results[tool_id] = tool.execute(function_params)
            
        for tool_id, result in results.items():
            observation_str += f"Tool: {tool_id}\n"
            observation_str += f"Result: {result}\n"
        observation_str += "</observation>"
        
        return observation_str
    
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        """
        This method is used to run the React operator.
        """
        
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
                "tool_call_id": "TOOL | tool_chooser"
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
                "tool_call_id": "TOOL | wikipedia"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.</thought>"
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"look_up\": {{\"keyword\": {{\"type\": \"string\", \"description\": \"Lookup the keywords in the given text.\"}}}}}}]",
                "tool_call_id": "TOOL | tool_chooser"
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
                "tool_call_id": "TOOL | look_up"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.</thought>"
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"look_up\": {{\"keyword\": {{\"type\": \"string\", \"description\": \"Lookup the keywords in the given text.\"}}}}}}]",
                "tool_call_id": "TOOL | tool_chooser"
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
                "tool_call_id": "TOOL | look_up"
            },
            {
                "role": "assistant",
                "content": "<thought>Thought 4: I need to instead search High Plains (United States).</thought>"
            },
            {
                "role": "tool",
                "content": "You can call the following functions: [{{\"wikipedia\": {{\"entity\": {{\"type\": \"string\", \"description\": \"The entity to search for in Wikipedia.\"}}}}}}]",
                "tool_call_id": "TOOL | tool_chooser"
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
                "tool_call_id": "TOOL | wikipedia"
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
        
        for i in range(self._max_iterations):
            thought_action = {
                'role': 'assistant',
                'content': f'<though>Thought {i}: '
            }
            thought_response = self._reasoning_llm.query(prompt = REACT_MESSAGE.append(thought_action), num_responses = 1, stop = [f"<action>"])
            thought_response_str = self._reasoning_llm.get_response_texts(query_responses = thought_response)[0]
            if 'finish' in thought_response_str.lower():
                break
            
            tool_chooser_response: Dict = self._tool_chooser.execute(input_message = thought_response_str)
            REACT_MESSAGE.append(tool_chooser_response)
            action = self._reasoning_llm.query(prompt = REACT_MESSAGE.append({'role': 'assistant', 'content': f'<action>Action {i}:'}), num_responses = 1, stop = [f"<observation>"])
            tool_calls_response = action['tool_calls']
            tool_observations: Dict = self._get_observation_by_executing_tool(input_message = tool_calls_response)
            REACT_MESSAGE.append(tool_observations)
        
        
        return AssistantMessagePrompt(input_message.prompt)