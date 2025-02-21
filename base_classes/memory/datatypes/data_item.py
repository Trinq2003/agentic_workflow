from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime

from base_classes.prompt import AbstractPrompt
from base_classes.system_component import SystemComponent
from base_classes.llm import AbstractLanguageModel
from base_classes.operator import AbstractOperator

class AbstractDataItem(ABC):
    _content: Any
    _created_timestamp: datetime
    _source: SystemComponent
    
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
    
    
class PromptDataItem(AbstractDataItem):
    _content: AbstractPrompt
    def __init__(self, content: AbstractPrompt, source: SystemComponent):
        super().__init__(content, source)

class OperatorDataItem(PromptDataItem):
    _source: AbstractOperator
    def __init__(self, content: AbstractPrompt, source: SystemComponent):
        super().__init__(content, source)
        
class LLMDataItem(PromptDataItem):
    _source: AbstractLanguageModel
    def __init__(self, content: AbstractPrompt, source: SystemComponent):
        super().__init__(content, source)
    