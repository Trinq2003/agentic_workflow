import uuid
import time
from typing import Iterable, List
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion, ChatCompletionMessage
from configuration.llm_inference_configuration import APILLMConfiguration
from abc import ABC, abstractmethod
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_xinference.chat_models import ChatXinference
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from base_classes.llm import AbstractLanguageModel
from base_classes.prompt import AbstractPrompt

class RequestLLM(AbstractLanguageModel):
    """
    RequestLLM is a concrete implementation of AbstractLanguageModel for querying various LLM models through LangChain.
    It supports multiple providers including OpenAI, Azure, HuggingFace, vLLM, Xinference, Ollama, DeepSeek, and Google.
    """
    _config: APILLMConfiguration = None
    def __init__(self, llm_config: APILLMConfiguration) -> None:
        """
        Initialize the RequestLLM with configuration and prepare the model.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        super().__init__(llm_config)

    def _load_model(self) -> None:
        """
        Load the language model using LangChain based on the configuration.
        """
        try:
            # Initialize callback manager for streaming if needed
            callback_manager = None
            if self._stream:
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            # Load model based on provider
            if self._config.model_provider == "azure":
                self._llm_model = AzureChatOpenAI(
                    azure_deployment=self._config.llm_api_deployment_name,
                    openai_api_version=self._config.llm_api_api_version,
                    azure_endpoint=self._config.llm_api_api_base,
                    api_key=self._config.llm_api_api_key,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            elif self._config.model_provider == "openai":
                self._llm_model = ChatOpenAI(
                    model_name=self._model_name,
                    openai_api_key=self._config.llm_api_api_key,
                    openai_api_base=self._config.llm_api_api_base,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            elif self._config.model_provider == "vllm":
                self._llm_model = ChatOpenAI(
                    model_name=self._model_name,
                    openai_api_key=self._config.llm_api_api_key,
                    openai_api_base=self._config.llm_api_api_base,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            elif self._config.model_provider == "xinference":
                self._llm_model = ChatXinference(
                    server_url=self._config.llm_api_api_base,
                    model_uid=self._model_name,
                    api_key=self._config.llm_api_api_key,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            elif self._config.model_provider == "ollama":
                self._llm_model = ChatOllama(
                    base_url=self._config.llm_api_api_base,
                    model=self._model_name,
                    temperature=self._temperature,
                    num_predict=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            elif self._config.model_provider == "deepseek":
                self._llm_model = ChatDeepSeek(
                    api_key=self._config.llm_api_api_key,
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            elif self._config.model_provider == "google":
                self._llm_model = ChatGoogleGenerativeAI(
                    google_api_key=self._config.llm_api_api_key,
                    model=self._model_name,
                    temperature=self._temperature,
                    max_output_tokens=self._max_tokens,
                    streaming=self._stream,
                    callback_manager=callback_manager
                )
            else:
                raise ValueError(f"Unsupported model provider: {self._config.model_provider}")

            self.logger.info(f"✅ Model loaded successfully: {self._model_name}")
        except Exception as e:
            self._llm_model = None
            self.logger.error(f"❌ Error loading model: {str(e)}")
            raise

    def _query(self, query: AbstractPrompt, num_responses: int = 1, stop: List[str] = None) -> ChatCompletion:
        """
        Query the language model using AbstractPrompt input format.

        :param query: The query to be posed to the language model.
        :type query: AbstractPrompt
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :param stop: List of stop sequences.
        :type stop: List[str]
        :return: The language model's response(s).
        :rtype: ChatCompletion
        """
        try:
            # Extract the prompt messages from AbstractPrompt
            prompt_messages = query.prompt
            
            # Convert OpenAI message format to LangChain message format
            messages = []
            for msg in prompt_messages:
                if msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            # Generate response
            response = self._llm_model.invoke(
                messages,
                stop=stop
            )
            self.logger.debug(f"Response: {response}")
            
            # Convert LangChain response to OpenAI ChatCompletion format
            return ChatCompletion.model_validate({
                "id": f"chatcmpl-{uuid.uuid4()}",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "created": int(time.time()),
                "model": self._model_name,
                "object": "chat.completion"
            })
        except Exception as e:
            self.logger.error(f"Error querying model: {str(e)}")
            return None
