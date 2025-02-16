from graph.node import InnerOperatorNode, OperatorNode, LLMNode, ToolNode
from base_classes.graph.edge import AbstractGraphEdge

class OperatorEdge(AbstractGraphEdge):
    """
    This class is used to represent an operator edge in the graph.
    """
    _start_node: OperatorNode
    _end_node: OperatorNode
    def __init__(self, start_node: OperatorNode, end_node: OperatorNode, edge_type: str, description: str = '') -> None:
        super().__init__(start_node = start_node, end_node = end_node, edge_type = edge_type, description = description)
        
class InnerOperatorEdge(AbstractGraphEdge):
    """
    This class is used to represent an inner operator edge in the graph.
    """
    _start_node: InnerOperatorNode
    _end_node: InnerOperatorNode
    def __init__(self, start_node: InnerOperatorNode, end_node: InnerOperatorNode, edge_type: str, description: str = '') -> None:
        super().__init__(start_node = start_node, end_node = end_node, edge_type = edge_type, description = description)