from typing import Any, Dict, List, Union

from base_classes.tool import AbstractTool
from configuration.tool_configuration import ToolConfiguration
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt

class DemonstrationSamplingTool(AbstractTool):
    """
    This class is used to demonstrate the sampling of the CoT operator.
    """
    def __init__(self, tool_config: ToolConfiguration) -> None:
        super().__init__(tool_config = tool_config)
        
    def _set_tool_data(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> None:
        self._data = {'message': input_message.text}
    
    def execute(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> List[Dict[str, Any]]:
        """
        This method is used to execute the demonstration sampling of the CoT operator.
        
        :return: A list of sample demonstrations (pre-defined QnA pairs).
        :rtype: List[Dict[str, Any]]
        """
        super().execute(input_message = input_message)