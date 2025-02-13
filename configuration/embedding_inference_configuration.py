from typing import List, Dict, Union, Any, Optional

from base_classes.configuration import Configuration

class EmbeddingModelConfiguration(Configuration):
    """
    Base class for Embedding Model Configurations.
    This includes shared properties for both API-based and locally deployed models.
    """
    model_model_name: str
    model_max_tokens: int
    model_embedding_dims: int
    model_identical_threshold: Optional[float]
    def __init__(self):
        super().__init__()
        self.sensitive_properties = []

    def _init_properties(self):
        """
        Define common properties for embedding models.
        """
        return [
            ['model.model_name', '', str], # Model ID or path
            ['model.max_tokens', 0, int], # Maximum tokens to generate
            ['model.embedding_dims', 0, int], # Number of dimensions in the embedding
            ['model.identical_threshold', 0.999, float], # Threshold for identical embeddings
        ]

class APIEmbeddingModelConfiguration(EmbeddingModelConfiguration):
    """
    Configuration for API-based embedding models.
    """
    emb_api_api_base: str
    emb_api_api_token: str
    emb_api_trust_remote_code: Optional[bool]
    cost_prompt_token_cost: float
    cost_response_token_cost: float
    retry_max_retries: int
    retry_backoff_factor: Optional[float]
    def __init__(self):
        super().__init__()
        
        sensitive_properties = ['emb_api.api_token']
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Extend base properties with API-specific configurations.
        """
        base_properties = super()._init_properties()
        additional_properties = [
            ['emb_api.api_base', '', str],
            ['emb_api.api_token', '', str],
            ['emb_api.trust_remote_code', False, bool],
            ['cost.prompt_token_cost', 0.0, float],
            ['cost.response_token_cost', 0.0, float],
            ['retry.max_retries', 3, int],
            ['retry.backoff_factor', 0.1, float],
        ]
        return base_properties + additional_properties
    
class LocalEmbeddingModelConfiguration(EmbeddingModelConfiguration):
    """
    Configuration for locally deployed embedding models.
    """
    path_local_model: str
    path_local_tokenizer: str
    path_local_config: Optional[str]
    path_local_cache: Optional[str]
    deployment_device: str
    deployment_tensor_parallel_size: Optional[int]
    deployment_gpu_memory_utilization: Optional[float]
    deployment_dtype: Optional[str]
    
    def __init__(self):
        super().__init__()
        
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Extend base properties with local model-specific configurations.
        """
        base_properties = super()._init_properties()
        additional_properties = [
            ['path.local_model', '', str], # Path to the local model
            ['path.local_tokenizer', '', str], # Path to the local tokenizer
            ['path.local_config', '', str],  # Local path to the model configuration
            ['path.local_cache', '', str],  # Local path to the cache
            ['deployment.device', 'cuda', str], # Device type (e.g., "cuda", "cpu")
            ['deployment.tensor_parallel_size', 1, int],  # Number of GPUs for tensor parallelism
            ['deployment.gpu_memory_utilization', 0.9, float],  # GPU memory utilization
            ['deployment.dtype', 'auto', str],  # Data type for model weights (e.g., "auto", "float16")
        ]
        return base_properties + additional_properties