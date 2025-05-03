from typing import List
import torch
import requests

from base_classes.embedding import AbstractEmbeddingModel
from configuration.embedding_inference_configuration import APIEmbeddingModelConfiguration


class HuggingfaceEmbeddingModel(AbstractEmbeddingModel):
    """
    The HuggingfaceEmbeddingModel class handles interactions with a Hugging Face deployed model.
    It inherits from AbstractEmbeddingModel and implements the necessary methods.
    """
    _config: APIEmbeddingModelConfiguration = None

    def __init__(self, embedding_model_config: APIEmbeddingModelConfiguration) -> None:
        """
        Initialize the HuggingfaceEmbeddingModel instance with configuration, model details, and caching options.
        """
        super().__init__(embedding_model_config)

        # Hugging Face specific parameters
        self.__hf_api_base: str = self._config.emb_api_api_base
        self.__hf_api_token: str = self._config.emb_api_api_token
        self.__hf_trust_remote_code: bool = self._config.emb_api_trust_remote_code
        self.__cost_prompt_token_cost: float = self._config.cost_prompt_token_cost
        self.__cost_response_token_cost: float = self._config.cost_response_token_cost
        self.__retry_max_retries: int = self._config.retry_max_retries
        self.__retry_backoff_factor: float = self._config.retry_backoff_factor

        self._load_model()

    def _load_model(self):
        """
        Load the Hugging Face inference model by verifying the API endpoint and token.
        """
        headers = {
            "Authorization": f"Bearer {self.__hf_api_token}"
        }
        response = requests.get(f"{self.__hf_api_base}", headers=headers)
        if response.status_code != 200:
            raise ValueError(f"❌ Failed to connect to Hugging Face embedding model: {response.status_code}, {response.text}")
        self.logger.info(f"✅ Hugging Face embedding model loaded successfully from {self.__hf_api_base}")
    
    def encode(self, text: str) -> List:
        """
        Generate embeddings for a given text using the Hugging Face model.

        :param text: Input text to embed.
        :return: List of floating point numbers representing the embedding.
        """
        headers = {
            "Authorization": f"Bearer {self.__hf_api_token}"
        }
        data = {"inputs": text}
        response = requests.post(
            f"{self.__hf_api_base}",
            headers=headers,
            json=data
        )
        if response.status_code != 200:
            raise ValueError(f"❌ Failed to get embedding: {response.status_code}, {response.text}")
        embedding = response.json()
        if not isinstance(embedding, list):
            raise ValueError(f"❌ Unexpected response format: {embedding}")
        return embedding
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts using the Hugging Face model.

        :param text1: First input text.
        :param text2: Second input text.
        :return: Similarity score between the two texts.
        """
        emb_text1 = self.encode(text1)
        emb_text2 = self.encode(text2)
        return float(torch.nn.functional.cosine_similarity(torch.tensor(emb_text1), torch.tensor(emb_text2)).numpy())