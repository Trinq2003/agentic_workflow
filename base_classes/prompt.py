from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageParam
from typing import Iterable

class AbstractPrompt(ABC):
    prompt: Iterable[ChatCompletionMessageParam] = None
    def __init__(self, prompt: Iterable[ChatCompletionMessageParam]) -> None:
        """
        Initialize the AbstractPrompt instance with the prompt.

        :param prompt: The prompt to be used for the completion.
        :type prompt: ChatCompletionMessageParam (OpenAI Compatible Prompt)
        """
        self.prompt = prompt