from typing import Any, Dict, List, Union

from base_classes.tool import AbstractTool
from configuration.tool_configuration import ToolConfiguration
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt
from graph.path import ToolPath

class PythonCodeRunnerTool(AbstractTool):
    """
    This class is used to decide which tool(s) to use for the a given prompt.
    """
    def __init__(self, tool_config: ToolConfiguration) -> None:
        super().__init__(tool_config = tool_config)
        
    def _set_tool_data(self, input_code_list: List[Dict[str, str]]) -> None:
        self._data = input_code_list
    
    def execute(self, input_code_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        This method is used to execute the tool chooser tool, which returns the tool ID to be used for the given prompt.
        
        :return: A graph of the tools to be used for the given prompt.
        :rtype: ToolPath
        """
        return super().execute(input_code_list = input_code_list)