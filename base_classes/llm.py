from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Iterable
import logging
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from openai import OpenAI

from configuration.llm_inference_configuration import LLMConfiguration

class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """
    _llm_model: OpenAI = None
    _config: LLMConfiguration = None
    _llm_id: str = None
    
    _model_name: str = None
    _temperature: float = None
    _max_tokens: int = None
    _cache: bool = None
    _cache_expiry: int = None
    _response_cache: Dict[str, List[Any]] = None
    prompt_tokens: int = None
    completion_tokens: int = None
    cost: float = None
    _query_call_count: int = None
    def __init__(self, llm_config: LLMConfiguration) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.load_config(llm_config)

        self._llm_id: str = self._config.llm_id
        self._model_name: str = self._config.model_model_name
        self._temperature: float = self._config.model_temperature
        self._max_tokens: int = self._config.model_max_tokens
        self._cache: bool = self._config.cache_enabled
        self._cache_expiry: int = self._config.cache_cache_expiry

        if self._cache:
            self._response_cache: Dict[str, List[Any]] = {}

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

        self._query_call_count: int = 0
        
    @property
    def llm_id(self) -> str:
        """
        Get the LLM ID.

        :return: The LLM ID.
        :rtype: str
        """
        return self._llm_id
    @property
    def temperature(self) -> float:
        """
        Get the sampling temperature.

        :return: The sampling temperature.
        :rtype: float
        """
        return self._temperature
    @temperature.setter
    def temperature(self, value: float) -> None:
        """
        Set the sampling temperature.

        :param value: The new sampling temperature.
        :type value: float
        """
        self._temperature = value
    
    def load_config(self, llm_config: LLMConfiguration) -> None:
        """
        Load a LLM configuration object.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self._config = llm_config
        self.logger.debug(f"Config loaded.")
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Abstract method to load the language model. Model config must be set before calling this method.
        """
        pass

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self._response_cache.clear()

    def _increment_chat_count(self) -> None:
        """
        Increment the chat call counter.
        """
        self._query_call_count += 1

    def get_query_call_count(self) -> int:
        """
        Get the number of times chat() has been called.

        :return: The chat call count.
        :rtype: int
        """
        return self._query_call_count
    
    def query(self, query: Iterable[ChatCompletionMessageParam], num_responses: int = 1) -> ChatCompletion:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        self._increment_chat_count()
        return self._query(query, num_responses)

    @abstractmethod
    def _query(self, query: Iterable[ChatCompletionMessageParam], num_responses: int = 1) -> ChatCompletion:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[ChatCompletion], ChatCompletion]) -> List[str]:
        """
        Abstract method to extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[Any], Any]
        :return: List of textual responses.
        :rtype: List[str]
        """
        pass
