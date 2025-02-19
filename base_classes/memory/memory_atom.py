from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Self, Literal
from functools import lru_cache

from base_classes.memory.management_term import MemoryState, AccessPermission, MemoryType
from base_classes.system_component import SystemComponent

@dataclass
class AbstractMemoryAtom(ABC):
    """Hierarchical memory unit combining architectural and agentic features"""
    _mem_atom_id: uuid.UUID # Globlally unique identifier
    _address: str # Relative address in the memory block
    _data: Any # Data stored in the memory atom
    _access_count: int
    _last_accessed: datetime
    _last_write: datetime
    _permission: Dict[str, AccessPermission] # Access permissions for different system components
    _state: MemoryState
    _required_atom: List[uuid.UUID] # List of memory atoms required for this atom to function
    _requiring_atom: List[uuid.UUID] # List of memory atoms requiring this atom to function
    _identifying_features: Any

    _list_of_mematom_ids: List[uuid.UUID] = []
    _mematom_instances_by_id: Dict[str, Self] = {}
    def __init__(self, address: int, data: Any, required_atom: List[uuid.UUID], requiring_atom: List[uuid.UUID]):
        """
        Represents the smallest indivisible unit of memory within a hierarchical memory system. 
        The data contained within an AbstractMemoryAtom is treated as an atomic entity from various perspectives.

        Example forms of memory atoms:
        - From the operator's inner perspective:
            - User's input prompt
            - Assistant's response
            - A thinking step
            - A tool execution process
        - From the system's perspective:
            - Operator response
        """
        self._mem_atom_id: uuid.UUID = uuid.uuid4()
        self._address: str = address
        self._data: Any = data
        self._required_atom: List[uuid.UUID] = required_atom
        self._requiring_atom: List[uuid.UUID] = requiring_atom
        self._access_count: int = 0
        self._last_accessed: datetime = datetime.now()
        self._last_write: datetime = self._last_accessed
        self._permission: Dict[str, AccessPermission] = {}
        self._state: MemoryState = MemoryState.USED
        
        self._init_identifying_features()
        
        if self._mem_atom_id in self.__class__._mematom_instances_by_id.keys():
            raise ValueError(f"❌ Memory Atom ID {self._mem_atom_id} is already initiated.")
        else:
            self.__class__._mematom_instances_by_id[self._mem_atom_id] = self

    @property
    def mem_atom_id(self):
        return self._mem_atom_id
    @property
    def address(self):
        return self._address
    @property
    def access_count(self):
        return self._access_count
    @property
    def last_accessed(self):
        return self._last_accessed
    @property
    def last_write(self):
        return self._last_write
    @property
    def permission(self):
        return self._permission
    @property
    def state(self):
        return self._state
    @property
    def required_atom(self):
        return self._required_atom
    @required_atom.setter
    def required_atom(self, required_atom: List[uuid.UUID]):
        self._required_atom = required_atom
    @property
    def requiring_atom(self):
        return self._requiring_atom
    @requiring_atom.setter
    def requiring_atom(self, requiring_atom: List[uuid.UUID]):
        self._requiring_atom = requiring_atom
    
    @classmethod
    def get_mematom_ids(cls) -> List[uuid.UUID]:
        """
        Get the list of memory atom IDs.

        :return: The list of memory atom IDs.
        :rtype: List[uuid.UUID]
        """
        return cls._list_of_mematom_ids
    @classmethod
    def get_mematom_instance_by_id(cls, mem_atom_id: uuid.UUID) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._mematom_instances_by_id.get(mem_atom_id, None)

    def _update_access(self):
        self._last_accessed = datetime.now()
        self._access_count += 1

    def read(self, requester: SystemComponent) -> Any:
        """Read data from the memory instance.
        """
        if self._check_permission(requester) > AccessPermission.NO_ACCESS:
            self._update_access()
            return self._data
        raise PermissionError(f"❌ {requester} lacks read permissions")

    def append_write(self, requester: SystemComponent, data: Any):
        """Append data to the memory instance without changing the existing data.

        Args:
            requester (Any): The system component requesting access (agent, tool, operator, ...).
            data (Any): Piece of data to append to the memory instance.

        Raises:
            PermissionError: If the requester lacks append writing permissions.
        """
        if self._check_permission(requester) >= AccessPermission.APPEND_ONLY:
            self._update_access()
            self._append_data(data)
        raise PermissionError(f"❌ {requester} lacks append writing permissions")
    
    def over_write(self, requester: SystemComponent, data: Any):
        """Overwrite existing data with the new one.

        Args:
            requester (Any): The system component requesting access (agent, tool, operator, ...).
            data (Any): Piece of data to append to the memory instance.

        Raises:
            PermissionError: If the requester lacks append writing permissions.
        """
        if self._check_permission(requester) >= AccessPermission.READ_WRITE:
            self._update_access()
            self._data = data
        raise PermissionError(f"❌ {requester} lacks overwriting permissions")
    
    def _check_permission(self, requester: SystemComponent) -> AccessPermission:
        """Return the access permission level for the requester for this memory instance.

        Args:
            requester (SystemComponent): The system component requesting access (agent, tool, operator, ...).
        """
        return self._permission.get(requester.component_id, AccessPermission.NO_ACCESS)
    
    @abstractmethod
    def _append_data(self, new_data: Any) -> None:
        """Append data to the memory instance. This is used in append writing mode.

        Args:
            new_data (Any): Piece of data to append to the memory instance.
        """
        pass
    
    @abstractmethod
    def _init_identifying_features(self) -> None:
        """
        Compile fast access distinctive identifying features from existing data 
        """
        pass
    
    @abstractmethod
    def _update_identifying_features(self) -> None:
        """
        Update identifying features with new data
        """
        