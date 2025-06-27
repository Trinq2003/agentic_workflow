from typing import List, Dict, Union, Any, Optional

from base_classes.configuration import Configuration

class OperatorConfiguration(Configuration):
    """
    Base configuration class for operator configurations.
    Contains common properties for both local and API-based deployments.
    """
    operator_operator_id: str
    operator_operator_type: str
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

class CoTOperatorConfiguration(OperatorConfiguration):
    """
    Configuration class for CoT operator.
    """
    cot_prompt_instruction: str
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]


    def _init_properties(self):
        """
        Define properties for CoT operator.
        """
        return [
            ['cot.prompt.instruction', '', str], # CoT prompt instruction
        ]

class ReActOperatorConfiguration(OperatorConfiguration):
    """
    Configuration class for CoT operator.
    """
    tool_tool_choser: str
    tool_callable: List[str]
    max_iterations: int
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]


    def _init_properties(self):
        """
        Define properties for CoT operator.
        """
        return [
            ['tool.tool_chooser', '', str], # Tool chooser
            ['tool.callable', [], list], # Callable tools
            ['max_iterations', 10, int], # Max iterations
        ]
        
class DebateOperatorConfiguration(OperatorConfiguration):
    """
    Configuration class for Debate operator.
    """
    debate_num_of_rounds: int
    debate_num_of_debaters: int
    debate_config_individual_configs: List[Dict[str, Any]]
    
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]

    def _init_properties(self):
        """
        Define properties for Debate operator.
        """
        return [
            ['debate.num_of_rounds', 5, int], # Number of rounds
            ['debate.num_of_debaters', 3, int], # Number of debaters
            ['debate.config.individual_configs', [], list], # Individual debater configurations
        ]
    
    def get_debater_configs(self) -> List[Dict[str, str]]:
        """
        Extract individual debater configurations from the loaded config.
        
        :return: List of debater configurations with llm_id
        :rtype: List[Dict[str, str]]
        """
        debater_configs = []
        self.logger.debug(f"Debater configs: {self.debate_config_individual_configs}")
        if hasattr(self, 'debate_config_individual_configs') and self.debate_config_individual_configs:
            for config_item in self.debate_config_individual_configs:
                if isinstance(config_item, dict) and 'debater' in config_item:
                    debater_configs.append(config_item['debater'])
        return debater_configs

class SelfConsistencyOperatorConfiguration(OperatorConfiguration):
    """
    Configuration class for Self-Consistency operator.
    """
    self_consistency_num_of_samples: int
    self_consistency_num_of_iterations: int
    self_consistency_config: Dict[str, Any]
    
    def __init__(self):
        super().__init__()
        sensitive_properties = []
        self.sensitive_properties = [property_.replace('.', '_') for property_ in sensitive_properties]


    def _init_properties(self):
        """
        Define properties for Self-Consistency operator.
        """
        return [
            ['self_consistency.num_of_samples', 3, int], # Number of samples (or reasoning paths)
            ['self_consistency.num_of_iterations', 2, int], # Number of iterations
        ]