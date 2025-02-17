from typing import List, Dict, Union, Any, Optional

from base_classes.configuration import Configuration

class OperatorConfiguration(Configuration):
    """
    Base configuration class for operator configurations.
    Contains common properties for both local and API-based deployments.
    """
    operator_id: str
    operator_type: str
    operator_enabled: bool
    operator_llm_component: List[str]
    operator_tool_component: List[str]
    execution_timeout: int
    execution_max_retry: int
    execution_backoff_factor: int
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]


    def _init_properties(self):
        """
        Define common properties for operator.
        """
        return [
            ['operator.operator_id', '', str], # Operator ID
            ['operator.operator_type', 'CoT', str], # Operator type
            ['operator.enabled', True, bool], # Operator enabled
            ['operator.llm_component', [], list], # LLM component
            ['operator.tool_component', [], list], # Tool component
            ['execution.timeout', 60, int], # Execution timeout
            ['execution.max_retry', 3, int], # Execution retry
            ['execution.backoff_factor', 2, int], # Execution backoff
        ]

class CoTOperatorConfiguration(Configuration):
    """
    Configuration class for CoT operator.
    """
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]


    def _init_properties(self):
        """
        Define properties for CoT operator.
        """
        return [
            ['operator.operator_id', '', str], # Operator ID
            ['operator.operator_type', 'CoT', str], # Operator type
            ['operator.enabled', True, bool], # Operator enabled
            ['operator.llm_component', [], list], # LLM component
            ['operator.tool_component', [], list], # Tool component
            ['execution.timeout', 60, int], # Execution timeout
            ['execution.max_retry', 3, int], # Execution retry
            ['execution.backoff_factor', 2, int], # Execution backoff
        ]

class ReActOperatorConfiguration(Configuration):
    """
    Configuration class for CoT operator.
    """
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]


    def _init_properties(self):
        """
        Define properties for CoT operator.
        """
        return [
            ['operator.operator_id', '', str], # Operator ID
            ['operator.operator_type', 'CoT', str], # Operator type
            ['operator.enabled', True, bool], # Operator enabled
            ['operator.llm_component', [], list], # LLM component
            ['operator.tool_component', [], list], # Tool component
            ['execution.timeout', 60, int], # Execution timeout
            ['execution.max_retry', 3, int], # Execution retry
            ['execution.backoff_factor', 2, int], # Execution backoff
        ]