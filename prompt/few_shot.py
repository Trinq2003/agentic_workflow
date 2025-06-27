from openai.types.chat import ChatCompletionMessageParam
from typing import List

from base_classes.prompt import AbstractPrompt

class FewShotPrompt(AbstractPrompt):
    """
    This class is used to define the Few-Shot Prompt.
    """
    prompt: List[ChatCompletionMessageParam] = None
    def __init__(self, prompt: List[ChatCompletionMessageParam]) -> None:
        super().__init__(prompt = prompt)