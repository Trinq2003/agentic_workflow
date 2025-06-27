from typing import Any, Dict, List, Union
import aiohttp

from base_classes.tool import AbstractTool
from configuration.tool_configuration import DemonstrationSamplingToolConfiguration
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.utils import parse_plan_xml

class DemonstrationSamplingTool(AbstractTool):
    """
    This class is used to demonstrate the sampling of the CoT operator.
    """
    def __init__(self, tool_config: DemonstrationSamplingToolConfiguration) -> None:
        super().__init__(tool_config = tool_config)
        # RAGFlow API specific configuration
        self._dataset_ids = getattr(tool_config, 'ragflow_dataset_ids', [])
        self._page = getattr(tool_config, 'ragflow_page', 1)
        self._page_size = getattr(tool_config, 'ragflow_page_size', 5)
        self._similarity_threshold = getattr(tool_config, 'ragflow_similarity_threshold', 0.2)
        self._vector_similarity_weight = getattr(tool_config, 'ragflow_vector_similarity_weight', 0.3)
        self._top_k = getattr(tool_config, 'ragflow_top_k', 1024)
        self._keyword = getattr(tool_config, 'ragflow_keyword', False)
        self._highlight = getattr(tool_config, 'ragflow_highlight', False)
    def _set_tool_data(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> None:
        """
        Set the tool data for RAGFlow API request.
        
        :param input_message: The input message containing the query
        :type input_message: Union[UserMessagePrompt, AssistantMessagePrompt]
        """
        # Extract the question from the input message
        question = input_message.text if hasattr(input_message, 'text') else str(input_message)
        
        # Prepare the request payload for RAGFlow API
        self._data = {
            "question": question,
            "dataset_ids": self._dataset_ids,
            "page": self._page,
            "page_size": self._page_size,
            "similarity_threshold": self._similarity_threshold,
            "vector_similarity_weight": self._vector_similarity_weight,
            "top_k": self._top_k,
            "keyword": self._keyword,
            "highlight": self._highlight
        }
    
    async def execute(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> List[str]:
        """
        Execute the RAGFlow API call to retrieve chunks based on the input message.
        
        :param input_message: The input message containing the query
        :type input_message: Union[UserMessagePrompt, AssistantMessagePrompt]
        :return: A list of retrieved chunks from RAGFlow API
        :rtype: List[str]
        """
        # Set the tool data
        self._set_tool_data(input_message)
        
        # Prepare the request URL and headers
        url = f"{self._webhook_base_url}/api/v1/retrieval"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self._headers_authorization}"
        }
        
        # try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=self._data) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Check if the API call was successful
                if result.get("code") == 0:
                    plans = []
                    chunks = result.get("data", {}).get("chunks", [])
                    return [chunk["content"] for chunk in chunks]
                else:
                    # Handle API error
                    error_message = result.get("message", "Unknown error")
                    self.logger.error(f"RAGFlow API error: {error_message}")
                    return []
                        
        # except aiohttp.ClientError as e:
        #     self.logger.error(f"Error calling RAGFlow API: {e}")
        #     return []
        # except Exception as e:
        #     self.logger.error(f"Unexpected error in RAGFlow API call: {e}")
        #     return []