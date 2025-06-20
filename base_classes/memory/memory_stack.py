import uuid
from typing import Dict, List, Self, Any

from base_classes.memory.memory_features import MemoryTopicFeature
from base_classes.memory.memory_topic import AbstractMemoryTopic
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.traceable_item import TimeTraceableItem
from base_classes.logger import HasLoggerClass

class AbstractMemoryStack(TimeTraceableItem, HasLoggerClass):
    _mem_stack_id: uuid.UUID
    _list_of_mem_topics: List[AbstractMemoryTopic]
    raw_context: str = ""
    refined_context: str = ""
    _sequence_of_mem_blocks: List[AbstractMemoryBlock]
    
    _memstack_instances_by_id: Dict[uuid.UUID, Self] = {}
    
    def __init__(self):
        super().__init__()
        self._mem_stack_id: uuid.UUID = uuid.uuid4()
        self._list_of_mem_topics: List[AbstractMemoryTopic] = []
        self._sequence_of_mem_blocks: List[AbstractMemoryBlock] = []
        if self._mem_stack_id in self.__class__._memstack_instances_by_id.keys():
            self.logger.error(f"❌ Memory Stack ID {self._mem_stack_id} is already initiated.")
            raise ValueError(f"❌ Memory Stack ID {self._mem_stack_id} is already initiated.")
        else:
            self.__class__._memstack_instances_by_id[self._mem_stack_id] = self

    @classmethod
    def get_memstack_ids(cls) -> List[uuid.UUID]:
        return cls._memstack_instances_by_id.keys()
    @classmethod
    def get_memstack_instance_by_id(cls, mem_stack_id: uuid.UUID) -> Self:
        return cls._memstack_instances_by_id[mem_stack_id]
    def add_mem_topic(self, mem_topic: AbstractMemoryTopic) -> None:
        assert mem_topic.mem_topic_id not in [topic.mem_topic_id for topic in self._list_of_mem_topics], "❌ Memory Topic ID already exists in Memory Stack."
        assert mem_topic.stack_container_id is None, "❌ Memory Topic already belongs to a Memory Stack."

        self._list_of_mem_topics.append(mem_topic)
        mem_topic.stack_container_id = self._mem_stack_id
    def get_address_of_topic_by_id(self, mem_topic_id: uuid.UUID) -> AbstractMemoryTopic:
        for index, mem_topic in enumerate(self._list_of_mem_topics):
            if mem_topic.mem_topic_id == mem_topic_id:
                return index
        self.logger.error(f"❌ Memory Topic ID {mem_topic_id} not found in Memory Stack {self._mem_stack_id}.")
        raise ValueError(f"❌ Memory Topic ID {mem_topic_id} not found in Memory Stack {self._mem_stack_id}.")
        
    @property
    def mem_stack_id(self) -> uuid.UUID:
        return self._mem_stack_id
    @property
    def list_of_mem_topics(self) -> List[AbstractMemoryTopic]:
        return self._list_of_mem_topics
    @list_of_mem_topics.setter
    def list_of_mem_topics(self, list_of_mem_topics: List[AbstractMemoryTopic]) -> None:
        self._list_of_mem_topics = list_of_mem_topics
    @property
    def sequence_of_mem_blocks(self) -> List[AbstractMemoryBlock]:
        return self._sequence_of_mem_blocks
    @sequence_of_mem_blocks.setter
    def sequence_of_mem_blocks(self, sequence_of_mem_blocks: List[AbstractMemoryBlock]) -> None:
        self._sequence_of_mem_blocks = sequence_of_mem_blocks