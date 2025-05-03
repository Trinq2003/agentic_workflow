import requests
from typing import Any, List, Union, Iterable
from openai import OpenAI, AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from configuration.llm_inference_configuration import APILLMConfiguration
from abc import ABC, abstractmethod

from base_classes.llm import AbstractLanguageModel

class AzureOpenAILLM(AbstractLanguageModel):
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
        Load the language model. This checks if the model API is reachable before initializing OpenAI.
        """
        # Construct the API health check request
        headers = {
            "Content-Type": "application/json",
            "api-key": self._config.llm_api_api_key
        }
        payload = {
            "messages": [{"role": "system", "content": "Health check"}],
            "max_tokens": 1
        }
        try:
            # Send a minimal POST request to check the service health
            response = requests.post(f"{self._config.llm_api_api_base}/openai/deployments/{self._config.model_model_name}/chat/completions?api-version={self._config.llm_api_api_version}", headers=headers, json=payload, timeout=5)

            if response.status_code == 200:
                self._llm_model = AzureOpenAI(
                    azure_endpoint=self._config.llm_api_api_base,
                    api_key=self._config.llm_api_api_key,
                    api_version=self._config.llm_api_api_version,
                    max_retries=self._config.retry_max_retries
                )
                self.logger.info(f"✅ Azure OpenAI model loaded successfully: {self._model_name}")
            else:
                self._llm_model = None
                self.logger.error(f"❌ Failed to load Azure OpenAI model, status code: {response.status_code}, Response: {response.text}")
        except requests.exceptions.RequestException as e:
            self._llm_model = None
            self.logger.error(f"❌ Error loading Azure OpenAI model: {str(e)}")

    def _query(self, query: Iterable[ChatCompletionMessageParam], num_responses: int = 1, stop: List[str] = ["\n"]) -> ChatCompletion:
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
            response = self._llm_model.chat.completions.create(
                model=self._model_name,
                messages=query,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                n=num_responses, # Number of responses to return
                stop=stop
            )
            
            return response  # Return the list of responses
        except Exception as e:
            self.logger.error(f"❌ Error querying Azure OpenAI model: {str(e)}")
            return None
