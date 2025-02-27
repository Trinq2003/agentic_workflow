from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageParam
from typing import List

class AbstractPrompt(ABC):
    prompt: List[ChatCompletionMessageParam] = None
    def __init__(self, prompt: List[ChatCompletionMessageParam]) -> None:
        """
        Initialize the AbstractPrompt instance with the prompt.

        :param prompt: The prompt to be used for the completion.
        :type prompt: ChatCompletionMessageParam (OpenAI Compatible Prompt)
        """
        self.prompt = prompt