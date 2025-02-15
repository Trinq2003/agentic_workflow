from openai.types.chat import ChatCompletionUserMessageParam
from typing import Iterable

from base_classes.prompt import AbstractPrompt

class UserMessagePrompt(AbstractPrompt):
    """
    User message prompt class.
    """
    _text: str = None
    def __init__(self, prompt: Iterable[ChatCompletionUserMessageParam]) -> None:
        super().__init__(prompt = prompt)
        assert len(self.prompt) == 1 and self.prompt[0].get("role") == "user"
        self._text = self.prompt[0].get("content")
        
    @property
    def text(self) -> str:
        """
        Get the text of the user message prompt.
        """
        return self._text