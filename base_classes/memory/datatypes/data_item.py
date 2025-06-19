from abc import ABC, abstractmethod
from typing import Any, Union
from datetime import datetime
import textwrap

from base_classes.prompt import AbstractPrompt
from base_classes.system_component import SystemComponent

class AbstractDataItem(ABC):
    _content: Any
    _source: Any
    _created_timestamp: datetime
    
    def __init__(self, content: Any, source: Any):
        self._content = content
        self._source = source
        self._created_timestamp = datetime.now()
    
    @property
    def content(self) -> Any:
        return self._content
    @property
    def created_timestamp(self) -> datetime:
        return self._created_timestamp
    @property
    def source(self) -> Any:
        return self._source    
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
class PromptDataItem(AbstractDataItem):
    _content: AbstractPrompt
    _source: Union[SystemComponent, str]
    def __init__(self, content: AbstractPrompt, source: Union[SystemComponent, str] = ""):
        super().__init__(content, source)
        
    def __str__(self) -> str:
        prompt = self._content.prompt
        formatted_messages = []
        for index, message in enumerate(prompt):
            role = message['role']
            content = message['content']
            
            # Handle case where content might be a dictionary or other non-string type
            if not isinstance(content, str):
                content = str(content)
            
            if role == "system": prefix = f"System message: \n"
            if role == "user": prefix = f"User message: \n"
            if role == "developer": prefix = f"Developer message: \n"
            if role == "assistant": prefix = f"Assistant response: \n"
            if role == "tool": prefix = f"Tool execution result: \n"
            formatted_messages.append(prefix + textwrap.indent(content, "\t"))
        
        return "\n".join(formatted_messages)
