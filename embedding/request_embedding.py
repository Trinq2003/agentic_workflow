from typing import List
import torch
import requests
import time
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, CohereEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import XinferenceEmbeddings

from base_classes.embedding import AbstractEmbeddingModel
from configuration.embedding_inference_configuration import APIEmbeddingModelConfiguration

class RequestEmbeddingModel(AbstractEmbeddingModel):
    """
    The RequestEmbeddingModel class handles interactions with various embedding models through LangChain.
    It inherits from AbstractEmbeddingModel and implements the necessary methods.
    """
    _config: APIEmbeddingModelConfiguration = None
    def __init__(self, embedding_model_config: APIEmbeddingModelConfiguration) -> None:
        """
        Initialize the RequestEmbeddingModel instance with configuration and prepare the model.
        """
        super().__init__(embedding_model_config)

        # Configuration parameters
        # self.logger.debug(f"List of APIEmbeddingModelConfiguration: {embedding_model_config.__dict__}")
        self.__emb_api_api_base: str = self._config.emb_api_api_base
        self.__emb_api_api_token: str = self._config.emb_api_api_token
        self.__emb_api_api_version: str = self._config.emb_api_api_version
        self.__emb_api_deployment_name: str = self._config.emb_api_deployment_name
        self.__emb_api_trust_remote_code: bool = self._config.emb_api_trust_remote_code
        self.__cost_prompt_token_cost: float = self._config.cost_prompt_token_cost
        self.__cost_response_token_cost: float = self._config.cost_response_token_cost
        self.__retry_max_retries: int = self._config.retry_max_retries
        self.__retry_backoff_factor: float = self._config.retry_backoff_factor

        self._load_model()

    def _load_model(self):
        """
        Load the embedding model using LangChain based on the configuration.
        """
        
        if self._config.model_provider == "azure":
            self._emb_model = AzureOpenAIEmbeddings(
                openai_api_version=self.__emb_api_api_version,
                openai_api_type="azure",
                api_key=self.__emb_api_api_token,
                azure_endpoint=self.__emb_api_api_base.rstrip("/"),
                azure_deployment=self.__emb_api_deployment_name,
                model=self._model_name
            )
        elif self._config.model_provider == "openai":
            self._emb_model = OpenAIEmbeddings(
                openai_api_key=self.__emb_api_api_token,
                openai_api_base=self.__emb_api_api_base,
                model=self._model_name
            )
        elif self._config.model_provider == "cohere":
            self._emb_model = CohereEmbeddings(
                cohere_api_key=self.__emb_api_api_token,
                model=self._model_name
            )
        elif self._config.model_provider == "xinference":
            self._emb_model = XinferenceEmbeddings(
                model_name=self._model_name,
                base_url=self.__emb_api_api_base,
                api_key=self.__emb_api_api_token
            )
        else:
            self._emb_model = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"trust_remote_code": self.__emb_api_trust_remote_code},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        self.logger.info(f"✅ Embedding model loaded successfully: {self._model_name}")
    
    def encode(self, text: str) -> List:
        """
        Generate embeddings for a given text using the LangChain model.

        :param text: Input text to embed.
        :return: List of floating point numbers representing the embedding.
        """
        for attempt in range(self.__retry_max_retries):
            try:
                embedding = self._emb_model.embed_query(text)
                # self.logger.debug(f"✅ Embedding generated for text '{text}'")
                return embedding
            except Exception as err:
                self.logger.warning(f"[❌ {self.__class__.__name__}] Error occurred: {err}. Retrying {attempt + 1}/{self.__retry_max_retries}...")
                time.sleep(self.__retry_backoff_factor * (2 ** attempt))  # Exponential backoff
        
        raise ValueError(f"[❌ {self.__class__.__name__}] Max retries exceeded. Failed to get embedding.")

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts using the embedding model.

        :param text1: First input text.
        :param text2: Second input text.
        :return: Similarity score between the two texts.
        """
        emb_text1 = self.encode(text1)
        emb_text2 = self.encode(text2)
        
        # Convert embeddings to tensors and add batch dimension
        emb1_tensor = torch.tensor(emb_text1).unsqueeze(0)  # Shape: [1, dim]
        emb2_tensor = torch.tensor(emb_text2).unsqueeze(0)  # Shape: [1, dim]
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1_tensor, emb2_tensor, dim=1)
        return float(similarity.item())