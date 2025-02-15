from base_classes.operator import AbstractOperator
from configuration.operator_configuration import OperatorConfiguration
from base_classes.tool import AbstractTool
from prompt.user_message import UserMessagePrompt
from prompt.few_shot import FewShotPrompt

class CoTOperator(AbstractOperator):
    """
    Inspired by Auto-Cot: Automatic Chain-of-Thought Prompting paper by AWS.
    Link: https://arxiv.org/pdf/2210.03493

    Args:
        AbstractOperator (_type_): _description_
    """
    _construct_cot_tool: AbstractTool = None
    def __init__(self, config: OperatorConfiguration) -> None:
        super().__init__(config = config)
        self._construct_cot_tool = self._tool_component[0]
        
    def demonstration_sampling(self, user_message: UserMessagePrompt) -> FewShotPrompt:
        """
        This method is used to demonstrate the sampling of the CoT operator.
        """
        pass