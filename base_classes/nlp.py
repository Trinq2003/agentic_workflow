from typing import Any, Dict, List, Optional, Tuple, Union, Self
from abc import ABC, abstractmethod

from base_classes.system_component import SystemComponent
from configuration.nlp_configuration import NLPConfiguration

class AbstractNLPModel(SystemComponent):
    """
    Abstract class for NLP models.
    """
    _nlp_model: Any = None
    _config: NLPConfiguration
    _nlp_model_id: str = None
    _nlp_model_by_id: Dict[str, Self] = {}
    
    _model_name: str = None

    def __init__(self, nlp_config: NLPConfiguration):
        super().__init__()
        self.load_config(nlp_config)
        self._nlp_model_id: str = "NLP | " + self._config.nlp_id
        self._model_name: str = self._config.model_name
        self.logger.debug(f"NLP model ID: {self._nlp_model_id}")
        
        if self._nlp_model_id in self.__class__._nlp_model_by_id.keys():
            self.logger.error(f"NLP Model ID {self._nlp_model_id} is already initiated.")
            raise ValueError(f"NLP Model ID {self._nlp_model_id} is already initiated.")
        else:
            self.__class__._nlp_model_by_id[self._nlp_model_id] = self
        
    def load_config(self, nlp_config: NLPConfiguration) -> None:
        """
        Load the configuration for the NLP model.
        """
        self._config = nlp_config
        
    @classmethod
    def get_emb_ids(cls) -> List[str]:
        return cls._nlp_model_by_id.keys()
    @classmethod
    def get_emb_instance_by_id(cls, nlp_id: str) -> Self:
        return cls._nlp_model_by_id.get(nlp_id, None)
    
    @property
    def nlp_model_id(self) -> str:
        """
        Get the NLP model ID.
        """
        return self._nlp_model_id
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text into words or sentences."""
        pass

    @abstractmethod
    def remove_stopwords(self, text: str) -> List[str]:
        """Remove stopwords from the tokenized text."""
        pass

    @abstractmethod
    def stem(self, text: str) -> List[str]:
        """Perform stemming on the tokenized text."""
        pass

    @abstractmethod
    def lemmatize(self, text: str) -> List[str]:
        """Perform lemmatization on the tokenized text."""
        pass

    @abstractmethod
    def vectorize(self, text: str) -> List[float]:
        """Convert the processed text into numerical vectors."""
        pass

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Summarize the input text."""
        pass

    @abstractmethod
    def pos_tagging(self, text: str) -> List[Tuple[str, str]]:
        """Perform Part-of-Speech tagging on the tokenized text."""
        pass

    @abstractmethod
    def word_segmentation(self, text: str) -> List[str]:
        """Segment text into words, especially useful for languages without spaces."""
        pass

    @abstractmethod
    def semantic_parsing(self, text: str) -> Dict[str, Any]:
        """Analyze the meaning of the text."""
        pass

    @abstractmethod
    def syntactic_parsing(self, text: str) -> Dict[str, Any]:
        """Parse the syntactic structure of the text."""
        pass

    @abstractmethod
    def lexical_parsing(self, text: str) -> Dict[str, Any]:
        """Analyze the lexical elements of the text."""
        pass

    @abstractmethod
    def extract_nouns(self, text: str) -> List[str]:
        """Extract nouns from the text."""
        pass

    @abstractmethod
    def extract_uuids(self, text: str) -> List[str]:
        """Extract UUIDs from the text."""
        pass
    
    @abstractmethod
    def extract_function_names(self, text: str) -> List[str]:
        """Extract function names from the text."""
        pass

    @abstractmethod
    def normalize_plural(self, word: str) -> str:
        """Normalize the plural form of a word."""
        pass
    
    @abstractmethod
    def extract_keywords(self, text: str) -> List[str]:
        """Extract comprehensive keywords from the text."""
        pass