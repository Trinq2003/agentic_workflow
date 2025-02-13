import requests
from typing import Any, List, Union, Iterable
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from configuration.llm_inference_configuration import APILLMConfiguration
from abc import ABC, abstractmethod

from base_classes.llm import AbstractLanguageModel

class RequestLLM(AbstractLanguageModel):
    """
    RequestLLM is a concrete implementation of AbstractLanguageModel for querying a VLLM model.
    It communicates with the model using HTTP requests.
    """
    _config: APILLMConfiguration = None
    def __init__(self, llm_config: APILLMConfiguration) -> None:
        """
        Initialize the RequestLLM with configuration and prepare the model request URL.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        super().__init__(llm_config)

    def _load_model(self) -> None:
        """
        Load the language model. In this case, we will just check if the model is reachable.
        """
        try:
            response = requests.get(f"{self._config.llm_api_api_base}/ping")
            if response.status_code == 200:
                self._llm_model: OpenAI = OpenAI(base_url=self._config.llm_api_api_base, api_key=self.llm_api_api_key, max_retries=self._config.retry_max_retries)
                self.logger.debug("Model loaded successfully.")
            else:
                self._llm_model = None
                self.logger.error(f"Failed to load model, status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")

    def _query(self, query: Iterable[ChatCompletionMessageParam], num_responses: int = 1) -> ChatCompletion:
        """
        Query the OpenAI language model using message-based input format.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        try:
            # Create the message list for the OpenAI API
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}]
            
            # OpenAI's GPT models that accept messages (e.g., gpt-3.5-turbo or gpt-4)
            response = self._llm_model.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                n=num_responses  # Number of responses to return
            )
            
            return response["choices"]  # Return the list of responses
        except Exception as e:
            self.logger.error(f"Error querying OpenAI model: {str(e)}")
            return None

    def get_response_texts(self, query_responses: Union[List[ChatCompletionMessageParam], ChatCompletion]) -> List[str]:
        """
        Extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[ChatCompletionMessageParam], ChatCompletion]
        :return: List of textual responses.
        :rtype: List[str]
        """
        if isinstance(query_responses, list):
            return [response["content"] for response in query_responses]  # Changed from "text" to "content"
        elif isinstance(query_responses, dict):
            return [query_responses.get("content", "")]
        return []
