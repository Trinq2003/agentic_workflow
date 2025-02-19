from abc import ABC, abstractmethod
import uuid
from typing import List, Dict, Self

from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.management_term import AccessPermission
from base_classes.system_component import SystemComponent

class AbstractMemory(ABC):
    _mem_id: uuid.UUID
    _memory_blocks: List[AbstractMemoryBlock] = []
    _memory_access_matrix: Dict[str, Dict[str, AccessPermission]] = {} # Access matrix for different system components per each memory block
    
    _list_of_memory_ids: List[uuid.UUID] = []
    _memory_instances_by_id: Dict[str, Self] = {}
    def __init__(self):
        self._mem_id: uuid.UUID = uuid.uuid4()
        self._memory_blocks: List[AbstractMemoryBlock] = []
        self._memory_access_matrix: Dict[str, Dict[str, AccessPermission]] = {}
        
        if self._mem_id in self.__class__._memory_instances_by_id.keys():
            raise ValueError(f"❌ Memory ID {self._mem_id} is already initiated.")
        else:
            self.__class__._memory_instances_by_id[self._mem_id] = self
            
    @property
    def mem_id(self) -> uuid.UUID:
        return self._mem_id
    @property
    def memory_blocks(self) -> List[AbstractMemoryBlock]:
        return self._memory_blocks
    @property
    def memory_access_matrix(self) -> Dict[str, Dict[str, AccessPermission]]:
        return self._memory_access_matrix
    
    @classmethod
    def get_memory_ids(cls) -> List[uuid.UUID]:
        return cls._list_of_memory_ids
    @classmethod
    def get_memory_instance_by_id(cls, mem_id: uuid.UUID) -> Self:
        return cls._memory_instances_by_id[mem_id]
            
    def add_memory_block(self, memory_block: AbstractMemoryBlock) -> None:
        if memory_block.mem_block_id in [mb_id.mem_block_id for mb_id in self._memory_blocks]:
            raise ValueError(f"❌ Memory Block with ID {memory_block.mem_block_id} had already existed in Memory {self._mem_id}.")
        else:
            self._memory_blocks.append(AbstractMemoryBlock.get_memblock_instance_by_id(memory_block.mem_block_id))
    
    @abstractmethod
    def _get_block_permission(self, requester: SystemComponent, memory_block: AbstractMemoryBlock) -> AccessPermission:
        pass
    
    @abstractmethod
    def get_memory_block(self, requester: SystemComponent, mem_block_id: uuid.UUID) -> AbstractMemoryBlock:
        pass
    
    @abstractmethod
    def get_memory_atom(self, requester: SystemComponent, mem_atom_id: uuid.UUID) -> AbstractMemoryAtom:
        pass