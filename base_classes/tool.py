from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Self
import logging

from configuration.tool_configuration import ToolConfiguration

class AbstractTool(ABC):
    """
    Abstract base class that defines the interface for all tools.
    """
    _config: ToolConfiguration = None
    _tool_id: str = None
    _list_of_tool_ids: List[str] = None
    _tool_instances_by_id: Dict[str, Self] = None
    
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
        
        self.__class__._list_of_tool_ids.append(self._tool_id)
        self.__class__._tool_instances_by_id[self._tool_id] = self
        
    @classmethod
    def get_tool_ids(cls) -> List[str]:
        """
        Get the list of operator IDs.

        :return: The list of operator IDs.
        :rtype: str
        """
        return cls._list_of_tool_ids
    @classmethod
    def get_tool_instance_by_id(cls, tool_id) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._tool_instances_by_id.get(tool_id, None)

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