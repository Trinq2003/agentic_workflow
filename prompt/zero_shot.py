from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from typing import Iterable, Union

from base_classes.prompt import AbstractPrompt

class ZeroShotPrompt(AbstractPrompt):
    """
    This class is used to define the Zero-Shot Prompt.
    """
    def __init__(self, prompt: Iterable[Union[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]]) -> None:
        super().__init__(prompt = prompt)
        assert len(self.prompt) == 1 and (self.prompt[0].get("role") == "system" or self.prompt[0].get("role") == "user")