import uuid
from typing import Dict, List, Self, Any

from base_classes.memory.memory_atom import AbstractMemoryAtom
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
    _block_address_in_memory: Dict[str, Any] = {}
    identifying_features: Dict[str, Any] = {}
    
    _list_of_memblock_ids: List[uuid.UUID] = []
    _memblock_instances_by_id: Dict[str, Self] = {}
    def __init__(self):
        self._mem_block_id: uuid.UUID = uuid.uuid4()
        self._memory_atoms: List[AbstractMemoryAtom] = []
        
        if self._mem_block_id in self.__class__._memblock_instances_by_id.keys():
            raise ValueError(f"❌ Memory Block ID {self._mem_block_id} is already initiated.")
        else:
            self.__class__._memblock_instances_by_id[self._mem_block_id] = self
    
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
    def block_address_in_memory(self) -> Dict[str, Any]:
        return self._block_address_in_memory
    @block_address_in_memory.setter
    def block_address_in_memory(self, block_address: Dict[str, Any]) -> None:
        self._block_address_in_memory = block_address
    
    @classmethod
    def get_memblock_ids(cls) -> List[uuid.UUID]:
        return cls._list_of_memblock_ids
    @classmethod
    def get_memblock_instance_by_id(cls, mem_block_id: uuid.UUID) -> Self:
        return cls._memblock_instances_by_id[mem_block_id]
    
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