from typing import Required, Optional, List

from base_classes.configuration import Configuration

class ToolConfiguration(Configuration):
    """
    Base configuration class for LLM models.
    Contains common properties for both local and API-based deployments.
    """
    tool_id: str
    webhook_base_url: str
    webhook_webhook_path: str
    webhook_method: Optional[str]
    headers_content_type: Optional[str]
    headers_authorization: str
    def __init__(self):
        super().__init__()

        # Define sensitive properties (if any)
        sensitive_properties = ['headers.authorization']
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Define common properties for all LLM configurations.
        """
        return [
            ['tool_id', '', str],  # ID of the tool
            ['webhook.base_url', '', str],  # Base URL for the webhook
            ['webhook.webhook_path', '', str],  # Path to the webhook
            ['webhook.method', '', str], # HTTP method for the webhook
            ['headers.content_type', 'application/json', str],  # Headers for the webhook
            ['headers.authorization', '', str],  # Authorization headers for the webhook
        ]

class DemonstrationSamplingToolConfiguration(ToolConfiguration):
    ragflow_dataset_ids: list[str]
    ragflow_page: int
    ragflow_page_size: int
    ragflow_similarity_threshold: float
    ragflow_vector_similarity_weight: float
    ragflow_top_k: int
    ragflow_keyword: bool
    ragflow_highlight: bool
    def __init__(self):
        super().__init__()
        
    def _init_properties(self):
        base_properties = super()._init_properties()
        ragflow_properties = [
            ['ragflow.dataset_ids', [], list[str]],
            ['ragflow.page', 1, int],
            ['ragflow.page_size', 5, int],
            ['ragflow.similarity_threshold', 0.2, float],
            ['ragflow.vector_similarity_weight', 0.3, float],
            ['ragflow.top_k', 1024, int],
            ['ragflow.keyword', False, bool],
            ['ragflow.highlight', False, bool],
        ]
        return base_properties + ragflow_properties