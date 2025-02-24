from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime

from base_classes.prompt import AbstractPrompt
from base_classes.system_component import SystemComponent
from base_classes.llm import AbstractLanguageModel
from base_classes.operator import AbstractOperator

class AbstractDataItem(ABC):
    _content: Any
    _source: SystemComponent
    _created_timestamp: datetime
    
    def __init__(self, content: Any, source: SystemComponent):
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
    def source(self) -> SystemComponent:
        return self._source    
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
class PromptDataItem(AbstractDataItem):
    _content: AbstractPrompt
    _source: SystemComponent
    def __init__(self, content: AbstractPrompt, source: SystemComponent):
        super().__init__(content, source)
        
    def __str__(self) -> str:
        prompt = self._content.prompt
        formatted_messages = []
        for index, message in enumerate(prompt):
            role = message.role
            content = message.content
            if role == "system": prefix = f"Message {index + 1}. System message: \n\t"
            if role == "user": prefix = f"Message {index + 1}. User message: \n\t"
            if role == "developer": prefix = f"Message {index + 1}. Developer message: \n\t"
            if role == "assistant": prefix = f"Message {index+1}. Assistant response: \n\t"
            if role == "tool": prefix = f"Message {index+1}. Tool execution result: \n\t"
            formatted_messages.append(prefix + content)
        
        return "\n".join(formatted_messages)

class OperatorDataItem(PromptDataItem):
    _content: AbstractPrompt
    _source: AbstractOperator
    def __init__(self, content: AbstractPrompt, source: AbstractOperator):
        super().__init__(content, source)
        
class LLMDataItem(PromptDataItem):
    _content: AbstractPrompt
    _source: AbstractLanguageModel
    def __init__(self, content: AbstractPrompt, source: AbstractLanguageModel):
        super().__init__(content, source)
    