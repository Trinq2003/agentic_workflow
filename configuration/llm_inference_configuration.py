from typing import Optional

from base_classes.configuration import Configuration

class LLMConfiguration(Configuration):
    """
    Base configuration class for LLM models.
    Contains common properties for both local and API-based deployments.
    """
    llm_id: str
    model_model_name: str
    model_temperature: float
    model_max_tokens: int
    cache_enabled: Optional[bool]
    cache_cache_expiry: Optional[int]
    def __init__(self):
        super().__init__()

        # Define sensitive properties (if any)
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Define common properties for all LLM configurations.
        """
        return [
            ['llm_id', '', str],  # ID of the LLM
            ['model.model_name', '', str],  # Model ID or path
            ['model.temperature', 1.0, float],  # Sampling temperature
            ['model.max_tokens', 0, int],  # Maximum tokens to generate
            ['cache.enabled', True, bool],  # Enable response caching
            ['cache.cache_expiry', 3600, int],  # Cache expiry time in seconds
        ]

class APILLMConfiguration(LLMConfiguration):
    """
    Configuration class for API-based LLM deployments (e.g., Azure OpenAI).
    Extends LLMConfiguration with API-specific properties.
    """
    llm_api_api_key: str
    llm_api_api_base: str
    llm_api_api_version: str
    llm_api_deployment_name: str
    cost_prompt_token_cost: float
    cost_response_token_cost: float
    retry_max_retries: int
    retry_backoff_factor: Optional[float]
    security_encrypted: Optional[bool]
    security_trust_strategy: Optional[str]
    def __init__(self):
        super().__init__()

        sensitive_properties = ['llm_api.api_key', 'llm_api.api_base']
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Extend the parent class's properties with API-specific properties.
        """
        base_properties = super()._init_properties()

        api_properties = [
            ['llm_api.api_key', '', str],  # API key for authentication
            ['llm_api.api_base', '', str],  # Base URL for the API
            ['llm_api.api_version', '', str],  # API version
            ['llm_api.deployment_name', '', str],  # Deployment name (e.g., Azure deployment ID)
            ['cost.prompt_token_cost', 0.0, float],  # Cost per prompt token
            ['cost.response_token_cost', 0.0, float],  # Cost per response token
            ['retry.max_retries', 5, int],  # Maximum number of retries
            ['retry.backoff_factor', 2, float],  # Backoff factor for retries
            ['security.encrypted', True, bool],  # Enable encryption
            ['security.trust_strategy', 'TRUST_ALL_CERTIFICATES', str],  # Trust strategy for certificates
        ]

        return base_properties + api_properties

class LocalLLMConfiguration(LLMConfiguration):
    """
    Configuration class for locally deployed language models (e.g., using VLLM).
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
        Extend LLMConfiguration properties with additional configurations for local LLM deployment.
        """
        base_properties = super()._init_properties()
        additional_properties = [
            ['path.local_model', '', str],  # Local path to the model
            ['path.local_tokenizer', '', str],  # Local path to the tokenizer
            ['path.local_config', '', str],  # Local path to the model configuration
            ['path.local_cache', '', str],  # Local path to the cache
            ['deployment.device', 'cuda', str], # Device type (e.g., "cuda", "cpu")
            ['deployment.tensor_parallel_size', 1, int],  # Number of GPUs for tensor parallelism
            ['deployment.gpu_memory_utilization', 0.9, float],  # GPU memory utilization
            ['deployment.dtype', 'auto', str],  # Data type for model weights (e.g., "auto", "float16")
        ]
        return base_properties + additional_properties