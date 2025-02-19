from abc import ABC, abstractmethod
import uuid
from typing import Dict, List, Self
from dataclasses import dataclass
from functools import lru_cache
import time

from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.management_term import AccessPermission
from base_classes.prompt import AbstractPrompt
from base_classes.system_component import SystemComponent

class AbstractMemoryBlock(ABC):
    """
    The AbstractMemoryBlock class represents a collection of AbstractMemoryAtom instances.
    It serves as a container for storing chains of conversations or actions, providing a structured 
    way to manage and access memory atoms.

    Each memory block can contain multiple memory atoms, facilitating the organization of 
    related data and interactions over time.
    """
    _mem_block_id: uuid.UUID
    _memory_atoms: List[AbstractMemoryAtom] = []
    _block_access_matrix: Dict[str, Dict[str, AccessPermission]] = {} # Access matrix for different system components per each memory atom
    _mem_atom_graph: Dict[uuid.UUID, List[uuid.UUID]] = {} # Graph of memory atoms and their dependencies
    
    _list_of_memblock_ids: List[uuid.UUID] = []
    _memblock_instances_by_id: Dict[str, Self] = {}
    def __init__(self):
        self._mem_block_id: uuid.UUID = uuid.uuid4()
        self._memory_atoms: List[AbstractMemoryAtom] = []
        self._block_access_matrix: Dict[str, Dict[str, AccessPermission]] = {}
        
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
    def block_access_matrix(self) -> Dict[str, Dict[str, AccessPermission]]:
        return self._block_access_matrix
    @property
    def mem_atom_graph(self) -> Dict[uuid.UUID, List[uuid.UUID]]:
        return self._mem_atom_graph
    
    @classmethod
    def get_memblock_ids(cls, mem_block_id: uuid.UUID) -> List[uuid.UUID]:
        return cls._list_of_memblock_ids
    @classmethod
    def get_memblock_instance_by_id(cls, mem_block_id: uuid.UUID) -> Self:
        return cls._memblock_instances_by_id[mem_block_id]
    
    
    def add_memory_atom(self, memory_atom: AbstractMemoryAtom) -> None:
        if memory_atom.mem_atom_id in [ma_id.mem_atom_id for ma_id in self._memory_atoms]:
            raise ValueError(f"❌ Memory Atom with ID {memory_atom.mem_atom_id} had already existed in Memory Block {self._mem_block_id}.")
        else:
            self._memory_atoms.append(AbstractMemoryAtom.get_mematom_instance_by_id(memory_atom.mem_atom_id))
    
        for required_atom_id in memory_atom.required_atom:
            if required_atom_id not in [ma_id.mem_atom_id for ma_id in self._memory_atoms]:
                self._memory_atoms.append(AbstractMemoryAtom.get_mematom_instance_by_id(required_atom_id))
                
            if required_atom_id not in self._mem_atom_graph[memory_atom.mem_atom_id]:
                self._mem_atom_graph[memory_atom.mem_atom_id].append(required_atom_id)
                
        for requiring_atom_id in memory_atom.requiring_atom:
            if requiring_atom_id not in [ma_id.mem_atom_id for ma_id in self._memory_atoms]:
                self._memory_atoms.append(AbstractMemoryAtom.get_mematom_instance_by_id(requiring_atom_id))
                
            if requiring_atom_id not in self._mem_atom_graph[memory_atom.mem_atom_id]:
                self._mem_atom_graph[memory_atom.mem_atom_id].append(requiring_atom_id)
    
    @abstractmethod
    def _get_permission(self, requester: SystemComponent, memory_atom: AbstractMemoryAtom) -> AccessPermission:
        pass
    
    @abstractmethod
    def get_memory_atom(self, requester: SystemComponent, mem_atom_id: uuid.UUID) -> AbstractMemoryAtom:
        pass
    
    @abstractmethod
    def _search_similar_memory_atom(self, query: AbstractPrompt, top_k = 3) -> List[AbstractMemoryAtom]:
        """Search for similar memory atoms in the memory block from the input query.

        Args:
            query (AbstractPrompt): This can be a prompt of user, assistant, tool response, etc.
            top_k (int, optional): The maximum number of returned items. Defaults to 3.

        Returns:
            List[AbstractMemoryAtom]: A list of memory atoms that are similar to the query.
        """
        pass
    
    def __len__(self) -> int:
        return len(self._memory_atoms)