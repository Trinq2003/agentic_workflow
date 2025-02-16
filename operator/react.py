from typing import Union

from base_classes.operator import AbstractOperator
from base_classes.tool import AbstractTool
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
    def __init__(self, config: ReActOperatorConfiguration) -> None:
        super().__init__(config = config)
        
    def _choose_tool_id(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> str:
        """
        This method is used to choose the tool for the React operator.
        :return: The tool ID for the React operator.
        """
        pass
    
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        """
        This method is used to run the React operator.
        """
        return AssistantMessagePrompt(input_message.prompt)