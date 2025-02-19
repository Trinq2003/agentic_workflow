from abc import ABC, abstractmethod

class Memory(ABC):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.banks: Dict[uuid.UUID, MemoryBank] = {}
            cls._instance.agent_registry: Dict[str, List[uuid.UUID]] = {}
            cls._instance.address_space: Dict[int, uuid.UUID] = {}
        return cls._instance