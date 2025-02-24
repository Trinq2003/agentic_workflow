from abc import abstractmethod
import uuid
from typing import List

from base_classes.llm import AbstractLanguageModel
from base_classes.embedding import AbstractEmbeddingModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from prompt.user_message import UserMessagePrompt
# from configuration.llm_inference_configuration import LLMConfiguration

class MemoryFeatureEngineer:
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
        
    def _generate_summary_from_memory_block(self, memory_block_id: uuid.UUID) -> str:
        """
        Generate a summary from a memory block.

        :param memory_block_id: The unique identifier of the memory block.
        :type memory_block_id: str
        :return: The summary of the memory block.
        :rtype: str
        """
        memory_block = AbstractMemoryBlock.get_memblock_instance_by_id(memory_block_id)
        memory_block_str = str(memory_block) # TODO: Add some prefix and suffix to this message, highlighting the aims and objectives of the summary.
        memory_block_summary_message = [
            {
                'role': 'user',
                'content': memory_block_str
            }
        ]
        memory_block_summary_prompt = UserMessagePrompt(prompt = memory_block_summary_message)
        raw_memory_block_summary = self._llm.query(memory_block_summary_prompt)
        memory_block_summary = self._llm.get_response_texts(raw_memory_block_summary)[0]
        
        memory_block.identifying_features['summary'] = memory_block_summary
    
    def _generate_embedding_for_memory_block_summary(self, memory_block_id: uuid.UUID) -> str:
        """
        Generate an embedding for the memory block summary.

        :param memory_block_id: The unique identifier of the memory block.
        :type memory_block_id: str
        :return: The embedding of the memory block summary.
        :rtype: str
        """
        memory_block = AbstractMemoryBlock.get_memblock_instance_by_id(memory_block_id)
        memory_block_summary: str = memory_block.identifying_features['summary']
        memory_block.identifying_features['summary_embedding'] = self._emb_model.encode(memory_block_summary)
        
    def memory_feature_engineering(self, memory_block_id: uuid.UUID) -> None:
        """
        Perform memory feature engineering on a memory block.

        :param memory_block_id: The unique identifier of the memory block.
        :type memory_block_id: str
        """
        self._generate_summary_from_memory_block(memory_block_id)
        self._generate_embedding_for_memory_block_summary(memory_block_id)