from typing import Dict, List

from graph.node import InnerOperatorNode, OperatorNode, LLMNode, ToolNode
from base_classes.graph.path import AbstractPath
from base_classes.tool import AbstractTool

class ToolPath(AbstractPath):
    """
    This class is used to represent a path of tools in the graph.
    """
    _used_tools: List[AbstractTool]
    def __init__(self, path_structure: Dict[str, List[str]]) -> None:
        super().__init__(path_structure = path_structure)
        
    def execute_tool_path(self):
        """
        Execute the path of tools.
        TODO: This is just executing all tools in the path. We need to implement the logic to execute the tools in the correct order.
        Furthermore, we want to parallize the execution of tools where possible.
        """
        for tuple_tool in self._path_structure.keys():
            tool = AbstractTool.get_tool_instance_by_id(tool_id = tuple_tool[1])
            tool.execute()