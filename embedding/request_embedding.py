from typing import List
import torch
import requests
import time

from base_classes.embedding import AbstractEmbeddingModel
from configuration.embedding_inference_configuration import APIEmbeddingModelConfiguration

class RequestEmbeddingModel(AbstractEmbeddingModel):
    """
    The HFEmbeddingModel class handles interactions with a Hugging Face deployed model.
    It inherits from AbstractEmbeddingModel and implements the necessary methods.
    """
    _config: APIEmbeddingModelConfiguration = None
    def __init__(self, embedding_model_config: APIEmbeddingModelConfiguration) -> None:
        """
        Initialize the HFEmbeddingModel instance with configuration, model details, and caching options.
        """
        super().__init__(embedding_model_config)

        # Hugging Face specific parameters
        self.logger.debug(f"List of APIEmbeddingModelConfiguration: {embedding_model_config.__dict__}")
        self.__emb_api_api_base: str = self._config.emb_api_api_base
        self.__emb_api_api_token: str = self._config.emb_api_api_token
        self.__emb_api_trust_remote_code: bool = self._config.emb_api_trust_remote_code
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
            "Authorization": f"Bearer {self.__emb_api_api_token}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.__emb_api_api_base}", headers=headers)
        if response.status_code != 200:
            
            raise ValueError(f"[❌ {self.__class__.__name__}] Failed to connect to API embedding model: {response.status_code}, {response.text}")
        self.logger.info(f"[✅ {self.__class__.__name__}] API embedding model loaded successfully from {self.__emb_api_api_base}")
    
    def encode(self, text: str) -> List:
        """
        Generate embeddings for a given text using the Hugging Face model.

        :param text: Input text to embed.
        :return: List of floating point numbers representing the embedding.
        """
        headers = {
            "Authorization": f"Bearer {self.__emb_api_api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": text
        }
        for attempt in range(self.__retry_max_retries):
            try:
                response = requests.post(
                    f"{self.__emb_api_api_base}",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f"[❌ {self.__class__.__name__}] Failed to get embedding: {response.status_code}, {response.text}")
                embedding = response.json()
                self.logger.debug(f"[✅ {self.__class__.__name__}] Embedding response for text \'{text}\': {embedding}")
                if not isinstance(embedding, list):
                    raise ValueError(f"[❌ {self.__class__.__name__}] Unexpected response format: {embedding}")
                return embedding
            except requests.exceptions.HTTPError as http_err:
                self.logger.warning(f"[❌ {self.__class__.__name__}] HTTP error occurred: {http_err}. Retrying {attempt + 1}/{self.__retry_max_retries}...")
            except requests.exceptions.RequestException as err:
                self.logger.warning(f"[❌ {self.__class__.__name__}] Error occurred: {err}. Retrying {attempt + 1}/{self.__retry_max_retries}...")
            time.sleep(self.__retry_backoff_factor * (2 ** attempt))  # Exponential backoff
        raise ValueError(f"[❌ {self.__class__.__name__}] Max retries exceeded. Failed to get embedding.")
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