from abc import abstractmethod
from typing import List, Dict
from torch import Tensor

from base_classes.llm import AbstractLanguageModel
from base_classes.embedding import AbstractEmbeddingModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.prompt import ICIOPrompt
from base_classes.memory.memory_topic import AbstractMemoryTopic
from base_classes.memory.management_term import MemoryBlockState

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
    
    # Memory block context refinement methods
    def refine_input_query(self, mem_block: AbstractMemoryBlock) -> None:
        list_of_relevant_mem_blocks = self.memory_block_retrieval(mem_block)
    
    # Memory block feature engineering methods
    def _extract_keywords_for_memory_block(self, mem_block: AbstractMemoryBlock) -> List[str]:
        """
        Extract important keywords from the input string using the LLM.

        Args:
            mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[str]: A list of extracted keywords.
        """
        keyword_dict = {}
        # Extract keywords for refined context
        self._keyword_extraction_prompt.context = mem_block.refined_input_query + "\n" + mem_block.refined_output_response
        raw_response = self._llm.query(str(self._keyword_extraction_prompt))
        keywords = [keyword.strip().lower() for keyword in self._llm.get_response_texts(raw_response)[0].split(",")]
        keyword_dict["refined"] = keywords
        # Extract keywords for raw context
        self._keyword_extraction_prompt.context = mem_block.input_query + "\n" + mem_block.output_response
        raw_response = self._llm.query(str(self._keyword_extraction_prompt))
        keywords = [keyword.strip().lower() for keyword in self._llm.get_response_texts(raw_response)[0].split(",")]
        keyword_dict["raw"] = keywords
        
        return keywords
    
    def _generate_embedding_for_memory_block(self, mem_block: AbstractMemoryBlock) -> List[Tensor] | Tensor:
        """
        Generate an embedding vector for the input string using the embedding model.

        Args:
            mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[Tensor] | Tensor: The generated embedding vector.
        """
        # Use the embedding model's encode method (assumes it returns List[float])
        return {
            'refined': {
                'refined_input_embedding': self._emb_model.encode(mem_block.refined_input_query),
                'refined_output_embedding': self._emb_model.encode(mem_block.refined_output_response)
            },
            'raw': {
                'input_embedding': self._emb_model.encode(mem_block.input_query),
                'output_embedding': self._emb_model.encode(mem_block.output_response),
                'context_embedding': self._emb_model.encode(str(mem_block))
            }
        }
    
    def feature_engineer_for_memory_block(self, mem_block: AbstractMemoryBlock) -> None:
        """
        Perform feature engineering on the memory block.

        Args:
            mem_block (AbstractMemoryBlock): The memory block to be processed.

        Returns:
            None
        """
        assert mem_block.mem_block_state == MemoryBlockState.REFINED_INPUT_AND_OUTPUT, "❌ Memory Block State should be REFINE."
        
        container_topic_id = mem_block.topic_container_id
        # Set the topic container ID for the memory block
        mem_block.identifying_features["address_in_topic"] = AbstractMemoryTopic.get_memtopic_instance_by_id(container_topic_id).get_address_of_block_by_id(mem_block.mem_block_id)
        
        keywords: Dict = self._extract_keywords_for_memory_block(mem_block)
        embedding_vector: Dict = self._generate_embedding_for_memory_block(mem_block)
        
        # Store the keyword features in the memory block
        mem_block.identifying_features["feature_for_raw_context"]["keywords"] = keywords["raw"]
        mem_block.identifying_features["feature_for_refined_context"]["keywords"] = keywords["refined"]
        
        # Store the embedding features in the memory block
        mem_block.identifying_features["feature_for_raw_context"]["context_embedding"] = embedding_vector["raw"]["context_embedding"]
        mem_block.identifying_features["feature_for_raw_context"]["input_embedding"] = embedding_vector["raw"]["input_embedding"]
        mem_block.identifying_features["feature_for_raw_context"]["output_embedding"] = embedding_vector["raw"]["output_embedding"]
        mem_block.identifying_features["feature_for_refined_context"]["refined_input_embedding"] = embedding_vector["refined"]["refined_input_embedding"]
        mem_block.identifying_features["feature_for_refined_context"]["refined_output_embedding"] = embedding_vector["refined"]["refined_output_embedding"]
        
        mem_block.mem_block_state = MemoryBlockState.FEATURE_ENGINEERED
    
    # Memory block retrieval method
    def memory_block_retrieval(self, mem_block: AbstractMemoryBlock) -> List[AbstractMemoryBlock]:
        pass