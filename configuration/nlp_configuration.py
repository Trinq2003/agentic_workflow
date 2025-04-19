from typing import Required, Optional

from base_classes.configuration import Configuration

class NLPConfiguration(Configuration):
    """
    Base configuration class for LLM models.
    Contains common properties for both local and API-based deployments.
    """
    nlp_id: str
    library: str
    model_name: str
    dependencies: Optional[list[str]]
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
            ['nlp_id', '', str],  # ID of the NLP model
            ['library', '', str],  # The library used for the NLP model (spacy, nltk, vn_core_nlp, etc.)
            ['model_name', '', str],  # Name of the NLP model (e.g., 'en_core
            ['dependencies', [], list]  # List of dependencies for the NLP model
        ]