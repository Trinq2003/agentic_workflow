from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageParam
from typing import List, Dict, Any, Optional

class PromptAtom(ChatCompletionMessageParam):
    """A class representing a single prompt message with ICIO components."""
    
    def __init__(
        self,
        instruction: str,
        context: str,
        input_indicator: str,
        output_indicator: str,
        role: str = "user",
        **kwargs: Any
    ) -> None:
        """
        Initialize a PromptAtom with ICIO components.

        :param instruction: The instruction part of the prompt
        :param context: The context part of the prompt
        :param input_indicator: Indicator for where input should be provided
        :param output_indicator: Indicator for expected output format
        :param role: The role of the message (e.g., "user", "system", "assistant")
        :param kwargs: Additional arguments for ChatCompletionMessageParam
        """
        # Construct the content using ICIO components
        content = f"<instruction>\n{instruction}\n</instruction>\n\n" \
                 f"<context>\n{context}\n</context>\n\n" \
                 f"<input_indicator>\n{input_indicator}\n</input_indicator>\n\n" \
                 f"<output_indicator>\n{output_indicator}\n</output_indicator>"
        
        # Initialize the parent ChatCompletionMessageParam
        super().__init__(content=content, role=role, **kwargs)
        
        # Store ICIO components as private attributes
        self._instruction = instruction
        self._context = context
        self._input_indicator = input_indicator
        self._output_indicator = output_indicator

    # Getters
    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def context(self) -> str:
        return self._context

    @property
    def input_indicator(self) -> str:
        return self._input_indicator

    @property
    def output_indicator(self) -> str:
        return self._output_indicator

    # Setters
    @instruction.setter
    def instruction(self, value: str) -> None:
        self._instruction = value
        self._update_content()

    @context.setter
    def context(self, value: str) -> None:
        self._context = value
        self._update_content()

    @input_indicator.setter
    def input_indicator(self, value: str) -> None:
        self._input_indicator = value
        self._update_content()

    @output_indicator.setter
    def output_indicator(self, value: str) -> None:
        self._output_indicator = value
        self._update_content()

    def _update_content(self) -> None:
        """Update the content attribute when any ICIO component changes."""
        self["content"] = f"<instruction>\n{self._instruction}\n</instruction>\n\n" \
                 f"<context>\n{self._context}\n</context>\n\n" \
                 f"<input_indicator>\n{self._input_indicator}\n</input_indicator>\n\n" \
                 f"<output_indicator>\n{self._output_indicator}\n</output_indicator>"


class AbstractPrompt(ABC):
    """Abstract base class for prompts composed of PromptAtom objects."""
    
    prompt: List[ChatCompletionMessageParam] = None
    
    def __init__(self, prompt: List[PromptAtom]) -> None:
        """
        Initialize the AbstractPrompt instance with a list of PromptAtom objects.

        :param prompt: List of PromptAtom objects forming the complete prompt
        :type prompt: List[PromptAtom]
        """
        self.prompt = prompt