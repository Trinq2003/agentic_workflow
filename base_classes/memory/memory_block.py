import uuid
from typing import Dict, List, Self, Any

from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_features import MemoryBlockFeature
from base_classes.memory.management_term import MemoryBlockState
from base_classes.prompt import AbstractPrompt
from base_classes.system_component import SystemComponent
from base_classes.traceable_item import TimeTraceableItem

class AbstractMemoryBlock(TimeTraceableItem):
    """
    The AbstractMemoryBlock class represents a collection of AbstractMemoryAtom instances.
    It serves as a container for storing chains of conversations or actions, providing a structured 
    way to manage and access memory atoms.

    Each memory block can contain multiple memory atoms, facilitating the organization of 
    related data and interactions over time.
    """
    _mem_block_id: uuid.UUID
    _memory_atoms: List[AbstractMemoryAtom] = []
    _mem_atom_graph: Dict[uuid.UUID, List[uuid.UUID]] = {} # Graph of memory atoms and their dependencies
    identifying_features: MemoryBlockFeature = {}
    
    _input_query: str = "" # Input query from the user or system
    _output_response: str = "" # Output response from the system or assistant
    _refined_input_query: str = "" # Refined input query after processing
    _refined_output_response: str = "" # Refined output response after processing
    _mem_block_state: MemoryBlockState
    _access_count: int = 0 # Access count for the memory block    
    _topic_container_id: uuid.UUID
    
    _memblock_instances_by_id: Dict[uuid.UUID, Self] = {}
    def __init__(self):
        self._mem_block_id: uuid.UUID = uuid.uuid4()
        self._memory_atoms: List[AbstractMemoryAtom] = []
        self._mem_block_state: MemoryBlockState = MemoryBlockState.EMPTY
        
        if self._mem_block_id in self.__class__._memblock_instances_by_id.keys():
            raise ValueError(f"❌ Memory Block ID {self._mem_block_id} is already initiated.")
        else:
            self.__class__._memblock_instances_by_id[self._mem_block_id] = self
    
    @classmethod
    def get_memblock_ids(cls) -> List[uuid.UUID]:
        """
        Get the list of memory block IDs.
        
        :return: The list of memory block IDs.
        :rtype: List[uuid.UUID]
        """
        return cls._memblock_instances_by_id.keys()
    @classmethod
    def get_memblock_instance_by_id(cls, mem_block_id: uuid.UUID) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._memblock_instances_by_id.get(mem_block_id, None)
    
    @property
    def mem_block_id(self) -> uuid.UUID:
        return self._mem_block_id
    @property
    def memory_atoms(self) -> List[AbstractMemoryAtom]:
        return self._memory_atoms
    @property
    def mem_atom_graph(self) -> Dict[uuid.UUID, List[uuid.UUID]]:
        return self._mem_atom_graph
    @mem_atom_graph.setter
    def mem_atom_graph(self, graph: Dict[uuid.UUID, List[uuid.UUID]]) -> None:
        self._mem_atom_graph = graph
        self._sync_dependencies()
    @property
    def input_query(self) -> str:
        self._access_count += 1
        return self._input_query
    @input_query.setter
    def input_query(self, query: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.RAW_INPUT_ONLY:
            self.mem_block_state = MemoryBlockState.RAW_INPUT_ONLY
        self._input_query = query
    @property
    def output_response(self) -> str:
        self._access_count += 1
        return self._output_response
    @output_response.setter
    def output_response(self, response: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.RAW_INPUT_AND_OUTPUT:
            self.mem_block_state = MemoryBlockState.RAW_INPUT_AND_OUTPUT
        self._output_response = response
    @property
    def refined_input_query(self) -> str:
        self._access_count += 1
        return self._refined_input_query
    @refined_input_query.setter
    def refined_input_query(self, query: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.REFINED_INPUT:
            self.mem_block_state = MemoryBlockState.REFINED_INPUT
        self._refined_input_query = query
    @property
    def refined_output_response(self) -> str:
        self._access_count += 1
        return self._refined_output_response
    @refined_output_response.setter
    def refined_output_response(self, response: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.REFINED_INPUT_AND_OUTPUT:
            self.mem_block_state = MemoryBlockState.REFINED_INPUT_AND_OUTPUT
        self._refined_output_response = response
    @property
    def topic_container_id(self) -> uuid.UUID:
        return self._topic_container_id
    @topic_container_id.setter
    def topic_container_id(self, topic_container_id: uuid.UUID) -> None:
        self._topic_container_id = topic_container_id
    @property
    def mem_block_state(self) -> MemoryBlockState:
        return self._mem_block_state
    @mem_block_state.setter
    def mem_block_state(self, state: MemoryBlockState) -> None:
        self._mem_block_state = state
    
    def add_memory_atom(self, memory_atom: AbstractMemoryAtom) -> None:
        self._add_one_node_without_dependencies(memory_atom)
    
        for required_atom_id in memory_atom.required_atom:
            self._add_one_node_without_dependencies(AbstractMemoryAtom.get_mematom_instance_by_id(required_atom_id))
            if required_atom_id not in self._mem_atom_graph[memory_atom.mem_atom_id]:
                self._mem_atom_graph[memory_atom.mem_atom_id].append(required_atom_id)
                
        for requiring_atom_id in memory_atom.requiring_atom:
            self._add_one_node_without_dependencies(AbstractMemoryAtom.get_mematom_instance_by_id(requiring_atom_id))
            if requiring_atom_id not in self._mem_atom_graph[memory_atom.mem_atom_id]:
                self._mem_atom_graph[memory_atom.mem_atom_id].append(requiring_atom_id)
                
        self._sync_dependencies()
    
    def _add_one_node_without_dependencies(self, memory_atom: AbstractMemoryAtom) -> None:
        if memory_atom.mem_atom_id in [ma_id.mem_atom_id for ma_id in self._memory_atoms]:
            raise ValueError(f"❌ Memory Atom with ID {memory_atom.mem_atom_id} had already existed in Memory Block {self._mem_block_id}.")
        else:
            self._memory_atoms.append(AbstractMemoryAtom.get_mematom_instance_by_id(memory_atom.mem_atom_id))
            self._mem_atom_graph[memory_atom.mem_atom_id] = []
    
    def __str__(self):
        # TODO: Rewrite the __str__ method to be more informative and be specialized for keyword extraction task.
        memory_block_str = []
        for memory_atom in self._memory_atoms:
            memory_atom_str = str(memory_atom)
            memory_block_str.append(memory_atom_str)
            
        prefix = f"Memory Block ID: {str(self._mem_block_id)}. This represents a chain of conversations or actions.\n"
        content = "The content of this memory block is as follows:\n\t" + '\n\t'.join(memory_block_str)
        suffix = ""
        
        return prefix + content + suffix
                
    def get_memory_atom(self, requester: SystemComponent, mem_atom_id: uuid.UUID) -> AbstractMemoryAtom:
        pass
    
    def _search_similar_memory_atom(self, query: AbstractPrompt, top_k = 3) -> List[AbstractMemoryAtom]:
        """Search for similar memory atoms in the memory block from the input query.

        Args:
            query (AbstractPrompt): This can be a prompt of user, assistant, tool response, etc.
            top_k (int, optional): The maximum number of returned items. Defaults to 3.

        Returns:
            List[AbstractMemoryAtom]: A list of memory atoms that are similar to the query.
        """
        pass
    
    def _sync_dependencies(self) -> None:
        """
        Synchronize the dependencies of memory atoms in the memory block.
        """
        for mem_atom_id in self._mem_atom_graph.keys():
            memory_atom = AbstractMemoryAtom.get_mematom_instance_by_id(mem_atom_id)
            memory_atom.required_atom = self._mem_atom_graph[mem_atom_id]
            for required_atom_id in memory_atom.required_atom:
                required_atom = AbstractMemoryAtom.get_mematom_instance_by_id(required_atom_id)
                new_requiring_atom = required_atom.requiring_atom + [mem_atom_id]
                required_atom.requiring_atom = new_requiring_atom
    
    def __len__(self) -> int:
        return len(self._memory_atoms)
    
class RealMemoryBlock(AbstractMemoryBlock):
    """
    The RealMemoryBlock class is a concrete implementation of the AbstractMemoryBlock class.
    It serves as a real storage for conversation between user and system.
    """
    pass

class SyntheticMemoryBlock(AbstractMemoryBlock):
    """
    The SyntheticMemoryBlock class is a concrete implementation of the AbstractMemoryBlock class.
    It serves as a refined storage based on the content of ReaslMemoryBlock (only keep the related content to the memory topic).
    """
    pass