from abc import abstractmethod
from typing import List
from torch import Tensor

from base_classes.llm import AbstractLanguageModel
from base_classes.embedding import AbstractEmbeddingModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.prompt import ICIOPrompt

class MemoryWorker:
    _llm: AbstractLanguageModel
    _emb_model: AbstractEmbeddingModel
    
    def __init__(self, llm: AbstractLanguageModel, emb_model: AbstractEmbeddingModel) -> None:
        """
        Initialize the Memory Language Model instance with configuration, model details, and caching options.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self._llm = llm
        self._emb_model = emb_model
        
        self._keyword_extraction_prompt = ICIOPrompt(
            instruction="Extract the most important keywords from the following text and list them separated by commas.",
            context="",
            input_indicator="",
            output_indicator="Go directly to the answer without any introduction. Output on one line of keywords, seperated by `,`.",
            role="user",
        )
    
    # Memory block feature engineering methods
    def _extract_keywords_for_memory_block(self, input_mem_block: AbstractMemoryBlock) -> List[str]:
        """
        Extract important keywords from the input string using the LLM.

        Args:
            input_mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[str]: A list of extracted keywords.
        """
        self._keyword_extraction_prompt.context = str(input_mem_block)
        raw_response = self._llm.query(str(self._keyword_extraction_prompt))
        keywords = [keyword.strip().lower() for keyword in self._llm.get_response_texts(raw_response)[0].split(",")]
        return keywords
    
    def _generate_embedding_for_memory_block(self, input_mem_block: AbstractMemoryBlock) -> List[Tensor] | Tensor:
        """
        Generate an embedding vector for the input string using the embedding model.

        Args:
            input_mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[Tensor] | Tensor: The generated embedding vector.
        """
        # Use the embedding model's encode method (assumes it returns List[float])
        return self._emb_model.encode(str(input_mem_block))
    
    def feature_engineer_for_memory_block(self, mem_block: AbstractMemoryBlock) -> None:
        """
        Perform feature engineering on the memory block.

        Args:
            mem_block (AbstractMemoryBlock): The memory block to be processed.

        Returns:
            None
        """
        keywords = self._extract_keywords_for_memory_block(mem_block)
        embedding_vector = self._generate_embedding_for_memory_block(mem_block)
        
        # Store the keywords and embedding vector in the memory block
        mem_block.identifying_features["keywords"] = keywords
        mem_block.identifying_features["embedding_vector"] = embedding_vector