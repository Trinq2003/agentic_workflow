from abc import ABC, abstractmethod
import uuid
from typing import List, Dict, Self, Any

from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.memory_feature_engineer import MemoryFeatureEngineer

class AbstractMemory(ABC):
    _mem_id: uuid.UUID
    _memory_blocks: Dict[uuid.UUID, AbstractMemoryBlock] = {}
    _memory_fe: MemoryFeatureEngineer
    
    _list_of_memory_ids: List[uuid.UUID] = []
    _memory_instances_by_id: Dict[str, Self] = {}
    def __init__(self):
        self._mem_id: uuid.UUID = uuid.uuid4()
        self._memory_blocks: Dict[uuid.UUID, AbstractMemoryBlock] = {}
        
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
    
    @classmethod
    def get_memory_ids(cls) -> List[uuid.UUID]:
        return cls._list_of_memory_ids
    @classmethod
    def get_memory_instance_by_id(cls, mem_id: uuid.UUID) -> Self:
        return cls._memory_instances_by_id[mem_id]
    
    def _create_memory_block_allocation_address(self) -> Dict[str, Any]:
        return {
            "memory_id": self._mem_id,
            "block_address": uuid.uuid4()
        }
            
    def add_memory_block(self, memory_block: AbstractMemoryBlock) -> None:
        if memory_block.mem_block_id in [mb_id for mb_id in self._memory_blocks.keys()]:
            raise ValueError(f"❌ Memory Block with ID {memory_block.mem_block_id} had already existed in Memory {self._mem_id}.")
        else:
            block_address = self._create_memory_block_allocation_address()
            memory_block.block_address_in_memory(block_address = block_address)
            self._memory_blocks[memory_block.mem_block_id] = AbstractMemoryBlock.get_memblock_instance_by_id(memory_block.mem_block_id)
            self._memory_fe.memory_feature_engineering(memory_block_id = memory_block.mem_block_id)
    
    def remove_memory_block(self, memory_block_id: uuid.UUID) -> None:
        if memory_block_id not in [mb_id for mb_id in self._memory_blocks.keys()]:
            raise ValueError(f"❌ Memory Block with ID {memory_block_id} does not exist in Memory {self._mem_id}.")
        else:
            del self._memory_blocks[memory_block_id]
                    
    def get_memory_block_by_id(self, mem_block_id: uuid.UUID) -> AbstractMemoryBlock:
        return self._memory_blocks[mem_block_id]