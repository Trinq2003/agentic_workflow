from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Self, Any, Union
import logging
import requests
import json

from configuration.tool_configuration import ToolConfiguration
from base_classes.system_component import SystemComponent

class AbstractTool(SystemComponent):
    """
    Abstract base class that defines the interface for all tools.
    """
    _config: ToolConfiguration = None
    _tool_id: str = None
    _list_of_tool_ids: List[str] = []
    _tool_instances_by_id: Dict[str, Self] = {}
    _data: Any = None
    
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
        self._webhook_method: str = self._config.webhook_method.upper()
        self._headers_content_type: str = self._config.headers_content_type
        self._headers_authorization: str = self._config.headers_authorization
        self._tool_id: str = self._config.tool_id
        
        if self._tool_id in self.__class__._list_of_tool_ids:
            raise ValueError(f"Tool ID {self._tool_id} is already initiated.")
        else:
            self.__class__._list_of_tool_ids.append(self._tool_id)
            
        if self._tool_id in self.__class__._tool_instances_by_id.keys():
            raise ValueError(f"Tool ID {self._tool_id} is already initiated.")
        else:
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

    def execute(self, **kwargs) -> Any:
        """
        Abstract method to execute the tool.
        """
        self._set_tool_data(**kwargs)
        
        url = f"{self._webhook_base_url}/{self._webhook_webhook_path}"
        headers = {"Content-Type": self._headers_content_type}
        
        if self._headers_authorization:
            headers["Authorization"] = self._headers_authorization
        
        try:
            if self._webhook_method == "POST":
                response = requests.post(url, headers=headers, data=json.dumps(self._data))
            elif self._webhook_method == "GET":
                response = requests.get(url, headers=headers, params=self._data)
            else:
                raise ValueError(f"Unsupported HTTP method: {self._webhook_method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling webhook: {e}")
            return None
    
    @abstractmethod
    def _set_tool_data(self, **kwargs) -> None:
        """
        Abstract method to set the tool data.
        Must be implemented by subclasses.
        """
        pass
    
    # @abstractmethod
    # def teardown(self) -> None:
    #     """
    #     Abstract method to clean up resources after execution.
    #     Must be implemented by subclasses.
    #     """
    #     pass