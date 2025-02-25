from base_classes.memory.memory import AbstractMemory

class OperatorMemory(AbstractMemory):
    """
    The OperatorMemory class represents the memory of an operator.
    It serves as a container for storing memory blocks, facilitating the organization of 
    related data and interactions over time.
    """
    def __init__(self) -> None:
        super().__init__()