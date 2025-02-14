from abc import ABC, abstractmethod
from typing import Optional
import logging

from configuration.tool_configuration import ToolConfiguration

class AbstractTool(ABC):
    """
    Abstract base class that defines the interface for all tools.
    """
    _config: ToolConfiguration = None
    _tool_id: str = None
    
    _webhook_base_url: str = None
    _webhook_webhook_path: str = None
    _webhook_method: Optional[str] = None
    _headers_content_type: Optional[str] = None
    _headers_authorization: str = None
    def __init__(self, tool_config: ToolConfiguration) -> None:
        """
        Initialize the AbstractTool instance with configuration.

        :param tool_config: The tool configuration object.
        :type tool_config: ToolConfiguration
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.load_config(tool_config)
        
        self._webhook_base_url: str = self._config.webhook_base_url
        self._webhook_webhook_path: str = self._config.webhook_webhook_path
        self._webhook_method: str = self._config.webhook_method
        self._headers_content_type: str = self._config.headers_content_type
        self._headers_authorization: str = self._config.headers_authorization
        self._tool_id: str = self._config.tool_id

    def load_config(self, tool_config: ToolConfiguration) -> None:
        """
        Load a tool configuration object.

        :param tool_config: The tool configuration object.
        :type tool_config: ToolConfiguration
        """
        self._config = tool_config
        self.logger.debug(f"Config loaded.")

    @abstractmethod
    def execute(self) -> None:
        """
        Abstract method to execute the tool.
        """
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """
        Abstract method to clean up resources after execution.
        Must be implemented by subclasses.
        """
        pass