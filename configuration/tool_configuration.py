from typing import Required, Optional

from base_classes.configuration import Configuration

class ToolConfiguration(Configuration):
    """
    Base configuration class for LLM models.
    Contains common properties for both local and API-based deployments.
    """
    webhook_base_url: str
    webhook_webhook_path: str
    webhook_method: Optional[str]
    headers_content_type: Optional[str]
    headers_authorization: str
    def __init__(self):
        super().__init__()

        # Define sensitive properties (if any)
        sensitive_properties = ['headers_authorization']
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Define common properties for all LLM configurations.
        """
        return [
            ['webhook.base_url', '', str],  # Base URL for the webhook
            ['webhook.webhook_path', '', str],  # Path to the webhook
            ['webhook.method', '', str] # HTTP method for the webhook
            ['headers.contenttype', 'application/json', str]  # Headers for the webhook
            ['headers.authorization', '', str]  # Authorization headers for the webhook
        ]
