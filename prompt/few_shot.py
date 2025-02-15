from openai.types.chat import ChatCompletionMessageParam
from typing import Iterable

from base_classes.prompt import AbstractPrompt

class FewShotPrompt(AbstractPrompt):
    """
    This class is used to define the Zero-Shot Prompt.
    """
    def __init__(self, prompt: Iterable[ChatCompletionMessageParam]) -> None:
        super().__init__(prompt = prompt)
        assert len(self.prompt) >= 3