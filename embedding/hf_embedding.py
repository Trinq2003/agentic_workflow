from typing import List
from sentence_transformers import SentenceTransformer
import torch

from base_classes.embedding import AbstractEmbeddingModel
from configuration.embedding_inference_configuration import LocalEmbeddingModelConfiguration

class HFLocalEmbeddingModel(AbstractEmbeddingModel):
    """
    The HFEmbeddingModel class handles interactions with a Hugging Face deployed model.
    It inherits from AbstractEmbeddingModel and implements the necessary methods.
    """
    _config: LocalEmbeddingModelConfiguration = None
    def __init__(self, embedding_model_config: LocalEmbeddingModelConfiguration) -> None:
        """
        Initialize the HFEmbeddingModel instance with configuration, model details, and caching options.
        """
        super().__init__(embedding_model_config)

        # Hugging Face specific parameters
        self.__model_local_path: str = self._config.path_local_model
        self.__tokenizer_local_path: str = self._config.path_local_tokenizer
        self.__config_local_path: str = self._config.path_local_config
        self.__cache_local_path: str = self._config.path_local_cache
        self.__device: str = self._config.deployment_device
        self.__tensor_parallel_size: int = self._config.deployment_tensor_parallel_size
        self.__gpu_memory_utilization: float = self._config.deployment_gpu_memory_utilization
        self.__dtype: str = self._config.deployment_dtype

        self._load_model()
            
    def _load_model(self):
        self._emb_model = SentenceTransformer(self.__model_local_path)

        # Set the specific device (GPU or CPU)
        if "cuda" in self.__device and torch.cuda.is_available():
            self._emb_model = self._emb_model.to(self.__device)
        elif self.__device == "cpu":
            self._emb_model = self._emb_model.to(torch.device("cpu"))
        else:
            raise ValueError(f"Unsupported device type or device not available: {self.__device}")
    
    def encode(self, text: str) -> List:
        """
        Generate embeddings for a given text using the Hugging Face model.

        :param text: Input text to embed.
        :return: List of floating point numbers representing the embedding.
        """
        return self._emb_model.encode(text)
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