from typing import Union, Dict, Any, List, Tuple
import json
from fastmcp import Client
import asyncio
from torch import Tensor
import uuid
import re
from copy import deepcopy

from base_classes.operator import AbstractOperator
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.datatypes.data_item import PromptDataItem
from configuration.operator_configuration import ReActOperatorConfiguration
from tools.tool_chooser import ToolChooserTool
from tools.demonstration_sampling import DemonstrationSamplingTool
from prompt.system_message import SystemMessagePrompt
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt
from prompt.tool_message import ToolMessagePrompt
from prompt.few_shot import FewShotPrompt
from base_classes.prompt import AbstractPrompt

class ReactOperator(AbstractOperator):
    """
    This operator is used to react to the user message.
    Operator design is inspired by ReAct: Synergizing Reasoning and Acting in Language Models paper by Shuyun Yao et al.
    Link: https://arxiv.org/abs/2210.03629.pdf
    """
    _config: ReActOperatorConfiguration = None
    _tool_chooser: ToolChooserTool = None
    _demonstration_sampling_tool: DemonstrationSamplingTool = None
    _mcps: List[Dict[str, Any]] = []
    _max_iterations: int = 10
    _reasoning_llm: AbstractLanguageModel = None
    _mcp_client_config: Dict[str, Any] = {}
    _callable_tools: List[Dict[str, str]] = []
    _tool_emb_dict: Dict[str, Tensor] = {}
    def __init__(self, config: ReActOperatorConfiguration) -> None:
        super().__init__(config = config)

        self._callable_tools = []
        self._max_iterations = config.react_tool_max_iterations
        self._mcps = config.react_mcps
        self._tool_chooser = self._tool_component[0]
        self._demonstration_sampling_tool = self._tool_component[1]
        self._reasoning_llm = self._llm_component[0] # Only 1 LLM component is allowed for React operator.

        self._mcp_client_config = {
            "mcpServers": {}
        }
        for mcp in self._mcps:
            self._mcp_client_config["mcpServers"][mcp['server_id']] = {
                "command": mcp['command'],
                "args": mcp['args']
            }

        self._callable_tools = asyncio.run(self.__connect_to_mcp_servers())
        self.logger.debug(f"Callable tools: {self._callable_tools}")
        self._tool_emb_dict = self._tool_chooser.get_tool_emb_dict(tool_description_dict = {tool['name']: tool['description'] for tool in self._callable_tools})
        self.logger.info(f"✅ Operator {self._operator_id} initiated successfully.")
        
    async def __connect_to_mcp_servers(self) -> List[Dict[str, str]]:
        lst_tools = []
        mcp_client = Client(self._mcp_client_config)
        async with mcp_client:
            tools = await mcp_client.list_tools()
            for tool in tools:
                lst_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                )
        return lst_tools

    def _choose_tool_id(self, input_message: AbstractPrompt) -> ToolMessagePrompt:
        """
        This method is used to choose the tool for the React operator. This method returns natural text for the tool to be used, in form of a prompt (tool call prompt).
        :return: The tool to be used for the React operator (natural text).
        """
        tools_to_call: List[Dict] = self._tool_chooser.execute(input_message = input_message, tools_dict=self._tool_emb_dict, top_k=self._config.react_tool_top_k) if self._tool_chooser else []
        self.logger.debug(f"Choosen tools: {tools_to_call}")
        # Only include schemas for the top recommended tools
        top_tool_names = {tool['tool_name'] for tool in tools_to_call}
        tool_schemas = []
        for tool in self._callable_tools:
            if tool["name"] in top_tool_names:
                tool_schema = {
                    "id": tool["name"],
                    "function": {
                        "arguments": tool["input_schema"]
                    }
                }
                tool_schemas.append(tool_schema)
        # Output both the top tools and their schemas
        return ToolMessagePrompt(prompt = [{
            "role": "tool",
            "content": (
                f"You can only call the following functions (with their call schema):\n"
                f"{json.dumps(tool_schemas, indent=2)}\n\n"
            ),
            "tool_call_id": "tool_chooser"
        }])
    
    async def _executing_tool(self, list_of_tools: List) -> List[ToolMessagePrompt]:
        mcp_client = Client(self._mcp_client_config)
        async with mcp_client:
            tool_results: List[ToolMessagePrompt] = []
            for tool in list_of_tools:
                self.logger.debug(f"Executing tool: {tool}")
                result = await mcp_client.call_tool(name = tool['id'], arguments = tool['function']['arguments'])
                tool_results.append(ToolMessagePrompt(prompt = [{'role': 'tool', 'content': str(result), 'tool_call_id': tool['id']}]))
            self.logger.debug(f"Tool results: {tool_results}")
            return tool_results
    
    async def _demonstration_sampling(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> FewShotPrompt:
        """
        This method is used to demonstrate the sampling of the ReAct operator.
        """

        demonstration_samples: List[Dict[str, Any]] = await self._demonstration_sampling_tool.execute(input_message = input_message)
        few_shot_demonstration_samples = FewShotPrompt(
            prompt = [
                {
                    "role": "tool",
                    "content": sample
                } for sample in demonstration_samples
            ]
        )
        self.logger.debug(f"Demonstration samples: {few_shot_demonstration_samples}")
        return few_shot_demonstration_samples
    
    def __extract_tool_calls_from_action(self, action_content: str):
        """
        Extracts tool call(s) from the <action>...</action> tag(s) in the LLM response.
        Returns a list of tool call dicts.
        """
        tool_calls = []
        # Find all <action>...</action> blocks
        action_blocks = re.findall(r"<action>(.*?)</action>", action_content, re.DOTALL)
        for block in action_blocks:
            block = block.strip()
            try:
                # Try to parse as JSON
                tool_call = json.loads(block)
                tool_calls.append(tool_call)
            except Exception as e:
                # If not valid JSON, skip or log
                self.logger.error(f"Failed to parse tool call: {block} ({e})")
        self.logger.debug(f"Tool calls after parsing: {tool_calls}")
        return tool_calls

    async def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> Tuple[AbstractPrompt, Dict[uuid.UUID, List[uuid.UUID]]]:
        """
        This method is used to run the React operator.
        """
        input_message_mem_atom = AbstractMemoryAtom(data = PromptDataItem(input_message))
        instruction_prompt = SystemMessagePrompt(prompt = [{"role": "system","content": self._config.react_prompt_instruction}])
        thought_prompt = SystemMessagePrompt(prompt = [{"role": "system","content": self._config.react_prompt_thought}]) + SystemMessagePrompt(prompt = [{"role": "system","content": f"The list of tools you can call are: {self._callable_tools}"}])
        action_prompt = SystemMessagePrompt(prompt = [{"role": "system","content": self._config.react_prompt_action}])
        instruction_prompt_mem_atom = AbstractMemoryAtom(data = PromptDataItem(instruction_prompt))
        demonstration_samples: FewShotPrompt = await self._demonstration_sampling(input_message = input_message)
        demonstration_samples_mem_atom = AbstractMemoryAtom(data = PromptDataItem(demonstration_samples))
        
        wrapped_input_prompt = UserMessagePrompt(prompt = [{"role": "user", "content": f"The user query is: \n {input_message.prompt[0].get('content')}."}])
        wrapped_input_prompt_mem_atom = AbstractMemoryAtom(data = PromptDataItem(wrapped_input_prompt))
        react_message = instruction_prompt + wrapped_input_prompt
        react_message_mem_atom = AbstractMemoryAtom(data = PromptDataItem(react_message))
        self.logger.debug(f"React message: {react_message.prompt}")

        dependency_graph: Dict[uuid.UUID, List[uuid.UUID]] = {}
        dependency_graph[input_message_mem_atom.mem_atom_id] = [demonstration_samples_mem_atom.mem_atom_id, wrapped_input_prompt_mem_atom.mem_atom_id]
        dependency_graph[instruction_prompt_mem_atom.mem_atom_id] = [react_message_mem_atom.mem_atom_id]
        dependency_graph[demonstration_samples_mem_atom.mem_atom_id] = [react_message_mem_atom.mem_atom_id]
        dependency_graph[wrapped_input_prompt_mem_atom.mem_atom_id] = [react_message_mem_atom.mem_atom_id]
        dependency_graph[react_message_mem_atom.mem_atom_id] = []

        react_mem_atoms: List[List[AbstractMemoryAtom]] = []
        for i in range(self._max_iterations):
            self.logger.debug(f"Iteration {i+1} of {self._max_iterations}")
            round_mem_atoms: List[AbstractMemoryAtom] = []
            # Thought step
            thought_prompt_input = deepcopy(react_message) + deepcopy(demonstration_samples) + deepcopy(thought_prompt)
            self.logger.debug(f"Thought prompt input: {thought_prompt_input.prompt}")
            raw_thought_response = self._reasoning_llm.query(query = thought_prompt_input, num_responses = 1, stop = [f"<action>"])
            self.logger.debug(f"Raw thought response: {raw_thought_response}")
            thought_response = AssistantMessagePrompt(prompt = [{'role': 'assistant', 'content': self._reasoning_llm.get_response_texts(query_responses = raw_thought_response)[0]}])
            react_message += thought_response
            thought_response_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(
                    content = thought_response,
                    source = self._reasoning_llm
                    )
                )
            round_mem_atoms.append(thought_response_mem_atom)
            if i == 0:
                dependency_graph[react_message_mem_atom.mem_atom_id] = [thought_response_mem_atom.mem_atom_id]
            else:
                dependency_graph[react_mem_atoms[-1][-1].mem_atom_id] = [thought_response_mem_atom.mem_atom_id]
            if '<finish>' in thought_response.prompt[0].get('content').lower():
                break
            # self.logger.debug(f"ReAct message after thought step: {react_message.prompt}")
            self.logger.debug(f"Completed thought step of iteration {i+1} of {self._max_iterations}")
            
            # Tool chooser step
            tool_chooser_response: ToolMessagePrompt = self._choose_tool_id(input_message = thought_response)
            tool_chooser_response_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(content = tool_chooser_response, source = self._tool_chooser)
            )
            round_mem_atoms.append(tool_chooser_response_mem_atom)
            dependency_graph[thought_response_mem_atom.mem_atom_id] = [tool_chooser_response_mem_atom.mem_atom_id]
            # self.logger.debug(f"ReAct message after tool chooser step: {react_message.prompt}")
            self.logger.debug(f"Completed tool chooser step of iteration {i+1} of {self._max_iterations}")
            
            # Action step
            action_prompt_input = deepcopy(react_message) + deepcopy(action_prompt) + deepcopy(tool_chooser_response) + deepcopy(wrapped_input_prompt)
            self.logger.debug(f"Action prompt input: {action_prompt_input.prompt}")
            action = self._reasoning_llm.query(query = action_prompt_input, num_responses = 1, stop = [f"<observation>"])
            self.logger.debug(f"Action: {action}")
            action_response = AssistantMessagePrompt(prompt = [{'role': 'assistant', 'content': self._reasoning_llm.get_response_texts(query_responses = action)[0]}])
            react_message += tool_chooser_response
            react_message += action_response
            action_mem_atom = AbstractMemoryAtom(
                data = PromptDataItem(content = action_response, source = self._reasoning_llm)
            )
            round_mem_atoms.append(action_mem_atom)
            dependency_graph[tool_chooser_response_mem_atom.mem_atom_id] = [action_mem_atom.mem_atom_id]
            dependency_graph[action_mem_atom.mem_atom_id] = []
            # self.logger.debug(f"ReAct message after action step: {react_message.prompt}")
            self.logger.debug(f"Completed action step of iteration {i+1} of {self._max_iterations}")
            
            # Tool execution step
            action_content = action.choices[0].message.content
            tool_calls_response = self.__extract_tool_calls_from_action(action_content)
            tool_execution_results: List[ToolMessagePrompt] = await self._executing_tool(list_of_tools = tool_calls_response)
            tool_observation_str = ""
            for tool_result in tool_execution_results:
                tool_observation_str += f"Tool: {tool_result.prompt[0].get('tool_call_id')}\n"
                tool_observation_str += f"\tResult: {tool_result.prompt[0].get('content')}\n"
                tool_execution_result_mem_atom = AbstractMemoryAtom(
                    data = PromptDataItem(content = tool_result, source = tool_result.prompt[0].get('tool_call_id'))
                )
                round_mem_atoms.append(tool_execution_result_mem_atom)
                dependency_graph[action_mem_atom.mem_atom_id].extend([tool_execution_result_mem_atom.mem_atom_id])
            react_message += ToolMessagePrompt(prompt = [{'role': 'tool', 'content': tool_observation_str, 'tool_call_id': 'environment'}])
            react_mem_atoms.append(round_mem_atoms)
            # self.logger.debug(f"ReAct message after tool execution step: {react_message.prompt}")
            self.logger.debug(f"Completed tool execution step of iteration {i+1} of {self._max_iterations}")
        return thought_response, dependency_graph