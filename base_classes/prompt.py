from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageParam
from typing import List, Dict, Any, Optional, Required, Union, Literal, TypedDict, TypeVar, Generic

from base_classes.logger import HasLoggerClass

class ICIOPrompt:
    """A class representing a single prompt message with ICIO components."""
    
    _instruction: str
    _context: str
    _input_indicator: str
    _output_indicator: str
    _role: str
    def __init__(
        self,
        instruction: str="",
        context: str="",
        input_indicator: str="",
        output_indicator: str="",
        role: str = "user",
    ) -> None:
        """
        Initialize a ICIOPrompt with ICIO components.

        :param instruction: The instruction part of the prompt
        :param context: The context part of the prompt
        :param input_indicator: Indicator for where input should be provided
        :param output_indicator: Indicator for expected output format
        :param role: The role of the message (e.g., "user", "system", "assistant")
        :param kwargs: Additional arguments for ChatCompletionMessageParam
        """
        
        # Store ICIO components as private attributes
        self._instruction = instruction
        self._context = context
        self._input_indicator = input_indicator
        self._output_indicator = output_indicator
        self._role = role
        
        # Construct the content using ICIO components
        self._update_content()

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
        
    @property
    def role(self) -> str:
        return self._role
        
    @property
    def content(self) -> str:
        return self._content

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
        out_string_prompt = ""
        if self._instruction:
            out_string_prompt += f"<instruction>\n{self._instruction}\n</instruction>\n\n"
        if self._context:
            out_string_prompt += f"<context>\n{self._context}\n</context>\n\n"
        if self._input_indicator:
            out_string_prompt += f"<input_indicator>\n{self._input_indicator}\n</input_indicator>\n\n"
        if self._output_indicator:
            out_string_prompt += f"<output_indicator>\n{self._output_indicator}\n</output_indicator>"
            
        self._content = out_string_prompt.strip()
                 
    def __str__(self) -> str:
        """Return a string representation of the ICIOPrompt."""
        return self._content
        
    def to_dict(self) -> Dict[str, str]:
        """Convert to a dictionary format compatible with ChatCompletionMessageParam."""
        return {
            "content": self._content,
            "role": self._role
        }


class AbstractPrompt(HasLoggerClass):
    """Abstract base class for prompts composed of ChatCompletionMessageParam objects."""
    
    prompt: List[ChatCompletionMessageParam] = None
    def __init__(self, prompt: List[ChatCompletionMessageParam]) -> None:
        """
        Initialize the AbstractPrompt instance with a list of ChatCompletionMessageParam objects.

        :param prompt: List of ChatCompletionMessageParam objects forming the complete prompt
        :type prompt: List[ChatCompletionMessageParam]
        """
        super().__init__()
        self.prompt = prompt
        # self.logger.debug(f"Prompt initialized with {len(self.prompt)} messages.")
    
    def __add__(self, other: "AbstractPrompt") -> "AbstractPrompt":
        """
        Add two prompts together.
        """
        return AbstractPrompt(prompt = self.prompt + other.prompt)
        