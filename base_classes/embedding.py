from abc import ABC, abstractmethod
from typing import List, Dict, Any, Self
import logging
from torch import Tensor

from configuration.embedding_inference_configuration import EmbeddingModelConfiguration
from base_classes.system_component import SystemComponent

class AbstractEmbeddingModel(SystemComponent):
    """
    Abstract base class that defines the interface for all embedding models.
    """
    _emb_moddel: Any = None
    _config: EmbeddingModelConfiguration = None
    _emb_id: str = None
    _emb_instances_by_id: Dict[str, Self] = {}
    
    _model_name: str = None
    _max_tokens: int = None
    _embedding_dims: int = None
    _identical_threshold: float = None
    def __init__(self, embedding_model_config: EmbeddingModelConfiguration) -> None:
        """
        Initialize the AbstractEmbeddingModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the embedding model. Defaults to "".
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_config(embedding_model_config)
        self._model_name: str = self._config.model_model_name
        self._max_tokens: int = self._config.model_max_tokens
        self._embedding_dims: int = self._config.model_embedding_dims
        self._emb_id: str = "EMBEDDING | " + self._config.emb_id

        self._identical_threshold: float = self._config.model_identical_threshold
        
        if self._emb_id in self.__class__._emb_instances_by_id.keys():
            raise ValueError(f"❌ Embedding ID {self._emb_id} is already initiated.")
        else:
            self.__class__._emb_instances_by_id[self._emb_id] = self

    @classmethod
    def get_emb_ids(cls) -> List[str]:
        """
        Get the list of embedding model IDs.

        :return: The list of embedding model IDs.
        :rtype: List[str]
        """
        return cls._emb_instances_by_id.keys()
    @classmethod
    def get_emb_instance_by_id(cls, emb_id: str) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._emb_instances_by_id.get(emb_id, None)

    @property
    def identical_threshold(self) -> float:
        """
        Get the identical threshold for similarity.

        :return: The identical threshold for similarity.
        :rtype: float
        """
        return self._identical_threshold
    
    @identical_threshold.setter
    def identical_threshold(self, threshold: float) -> None:
        """
        Set the identical threshold for similarity.

        :param threshold: The identical threshold for similarity.
        :type threshold: float
        """
        self._identical_threshold = threshold
    
    def load_config(self, embedding_model_config: EmbeddingModelConfiguration) -> None:
        """
        Load a EmbeddingModel configuration object.

        :param embedding_config_obj: The EmbeddingModel configuration object.
        :type embedding_config_obj: EmbeddingModelConfiguration
        """
        self._config = embedding_model_config

        self.logger.debug(f"Config loaded.")
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the embedding model.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[Tensor] | Tensor:
        """
        Abstract method to generate embeddings for a given text.

        :param text: Input text to embed.
        :return: List of floating point numbers representing the embedding.
        """
        pass
    
    @abstractmethod
    def similarity(self, text1: str, text2: str) -> float:
        """
        Abstract method to calculate the similarity between two texts.

        :param text1: First input text.
        :param text2: Second input text.
        :return: Similarity score between the two texts.
        """
        pass
    
    def is_identical(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are identical.

        :param text1: First input text.
        :param text2: Second input text.
        :return: True if the texts are identical, False otherwise.
        """
        return self.similarity(text1, text2) >= self.identical_threshold