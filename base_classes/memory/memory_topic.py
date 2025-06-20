import uuid
from typing import Dict, List, Self, Any
import numpy as np
import torch

from base_classes.memory.memory_features import MemoryTopicFeature
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.traceable_item import TimeTraceableItem
from base_classes.logger import HasLoggerClass
from base_classes.memory.memory_block import MemoryBlockState

class AbstractMemoryTopic(TimeTraceableItem, HasLoggerClass):
    _mem_topic_id: uuid.UUID
    _chain_of_memblocks: List[AbstractMemoryBlock]
    identifying_features: MemoryTopicFeature = {}
    raw_context: str = ""
    refined_context: str = ""
    _stack_container_id: uuid.UUID = None
    
    _memtopic_instances_by_id: Dict[uuid.UUID, Self] = {}
    
    def __init__(self):
        HasLoggerClass.__init__(self)
        TimeTraceableItem.__init__(self)
        self._mem_topic_id: uuid.UUID = uuid.uuid4()
        self._chain_of_memblocks: List[AbstractMemoryBlock] = []
        self.identifying_features: MemoryTopicFeature = {
            "keywords": [],
            "embedding": None
        }
        
        if self._mem_topic_id in self.__class__._memtopic_instances_by_id.keys():
            self.logger.error(f"❌ Memory Topic ID {self._mem_topic_id} is already initiated.")
            raise ValueError(f"❌ Memory Topic ID {self._mem_topic_id} is already initiated.")
        else:
            self.__class__._memtopic_instances_by_id[self._mem_topic_id] = self

    @classmethod
    def get_memtopic_ids(cls) -> List[uuid.UUID]:
        return cls._memtopic_instances_by_id.keys()
    @classmethod
    def get_memtopic_instance_by_id(cls, mem_topic_id: uuid.UUID) -> Self:
        return cls._memtopic_instances_by_id[mem_topic_id]
    
    def add_mem_block(self, mem_block: AbstractMemoryBlock) -> None:
        from base_classes.memory.memory_stack import AbstractMemoryStack
        
        assert self._stack_container_id is not None, "❌ Memory Topic does not belong to any memory stack."
        assert mem_block.mem_block_state > MemoryBlockState.EMPTY, "❌ Memory Block is empty."
        self._chain_of_memblocks.append(mem_block)
        mem_block.topic_container_ids.append(self._mem_topic_id)

        mem_stack = AbstractMemoryStack.get_memstack_instance_by_id(self._stack_container_id)
        if len(mem_stack.sequence_of_mem_blocks) == 0:
            mem_stack.sequence_of_mem_blocks.append(mem_block)
        else:
            if mem_stack.sequence_of_mem_blocks[-1].mem_block_id != mem_block.mem_block_id: 
                mem_stack.sequence_of_mem_blocks.append(mem_block)
        self.logger.debug(f"Inserted Memory Block ID {mem_block.mem_block_id} into Memory Topic {self._mem_topic_id}. New MemBlock's topic conatiner ID: {mem_block.topic_container_ids}.")
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