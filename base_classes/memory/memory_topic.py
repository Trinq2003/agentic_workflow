import uuid
from typing import Dict, List, Self, Any

from base_classes.memory.memory_features import MemoryTopicFeature
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.traceable_item import TimeTraceableItem

class AbstractMemoryTopic(TimeTraceableItem):
    _mem_topic_id: uuid.UUID
    _chain_of_memblocks: List[AbstractMemoryBlock]
    identifying_features: MemoryTopicFeature = {}
    raw_context: str = ""
    refined_context: str = ""
    _stack_container_id: uuid.UUID
    
    _memtopic_instances_by_id: Dict[uuid.UUID, Self] = {}
    
    def __init__(self):
        self._mem_topic_id: uuid.UUID = uuid.uuid4()
        self._chain_of_memblocks: List[AbstractMemoryBlock] = []
        
        if self._mem_topic_id in self.__class__._memtopic_instances_by_id.keys():
            raise ValueError(f"❌ Memory Topic ID {self._mem_topic_id} is already initiated.")
        else:
            self.__class__._memtopic_instances_by_id[self._mem_topic_id] = self

    @classmethod
    def get_memtopic_ids(cls) -> List[uuid.UUID]:
        return cls._memtopic_instances_by_id.keys()
    @classmethod
    def get_memtopic_instance_by_id(cls, mem_topic_id: uuid.UUID) -> Self:
        return cls._memtopic_instances_by_id[mem_topic_id]
    
    def insert_mem_block(self, mem_block: AbstractMemoryBlock) -> None:
        self._chain_of_memblocks.append(mem_block)
        mem_block.topic_container_id(self._mem_topic_id)
    def get_address_of_block_by_id(self, mem_block_id: uuid.UUID) -> int:
        for index, mem_block in enumerate(self._chain_of_memblocks):
            if mem_block.mem_block_id == mem_block_id:
                return index
        raise ValueError(f"❌ Memory Block ID {mem_block_id} not found in Memory Topic {self._mem_topic_id}.")
    @property
    def mem_topic_id(self) -> uuid.UUID:
        return self._mem_topic_id
    @property
    def chain_of_memblocks(self) -> List[AbstractMemoryBlock]:
        return self._chain_of_memblocks
    @property
    def stack_container_id(self) -> uuid.UUID:
        return self._stack_container_id
    @stack_container_id.setter
    def stack_container_id(self, stack_container_id: uuid.UUID) -> None:
        self._stack_container_id = stack_container_id