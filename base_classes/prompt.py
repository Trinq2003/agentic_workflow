from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageParam
from typing import List, Dict, Any, Optional, Required, Union, Literal, TypedDict, TypeVar, Generic

class ICIOPrompt(TypedDict, total=False):
    """A class representing a single prompt message with ICIO components."""
    content: Required[str]
    role: Required[Literal["user", "developer", "assistant", "system", "tool"]]
    
    _instruction: str
    _context: str
    _input_indicator: str
    _output_indicator: str
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
        # Construct the content using ICIO components
        content = ""
        if instruction:
            content += f"<instruction>\n{instruction}\n</instruction>\n\n"
        if context:
            content += f"<context>\n{context}\n</context>\n\n"
        if input_indicator:
            content += f"<input_indicator>\n{input_indicator}\n</input_indicator>\n\n"
        if output_indicator:
            content += f"<output_indicator>\n{output_indicator}\n</output_indicator>"
        
        # Initialize the parent ChatCompletionMessageParam
        self.content = content.strip()
        self.role = role
        
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
        out_string_prompt = ""
        if self._instruction:
            out_string_prompt += f"<instruction>\n{self._instruction}\n</instruction>\n\n"
        if self._context:
            out_string_prompt += f"<context>\n{self._context}\n</context>\n\n"
        if self._input_indicator:
            out_string_prompt += f"<input_indicator>\n{self._input_indicator}\n</input_indicator>\n\n"
        if self._output_indicator:
            out_string_prompt += f"<output_indicator>\n{self._output_indicator}\n</output_indicator>"
            
        self["content"] = out_string_prompt.strip()
                 
    def __str__(self) -> str:
        """Return a string representation of the ICIOPrompt."""
        out_string_prompt = ""
        if self._instruction:
            out_string_prompt += f"<instruction>\n{self._instruction}\n</instruction>\n\n"
        if self._context:
            out_string_prompt += f"<context>\n{self._context}\n</context>\n\n"
        if self._input_indicator:
            out_string_prompt += f"<input_indicator>\n{self._input_indicator}\n</input_indicator>\n\n"
        if self._output_indicator:
            out_string_prompt += f"<output_indicator>\n{self._output_indicator}\n</output_indicator>"
        return out_string_prompt.strip()


class AbstractPrompt(ABC):
    """Abstract base class for prompts composed of ChatCompletionMessageParam objects."""
    
    prompt: List[ChatCompletionMessageParam] = None
    
    def __init__(self, prompt: List[ChatCompletionMessageParam]) -> None:
        """
        Initialize the AbstractPrompt instance with a list of ChatCompletionMessageParam objects.

        :param prompt: List of ChatCompletionMessageParam objects forming the complete prompt
        :type prompt: List[ChatCompletionMessageParam]
        """
        self.prompt = prompt