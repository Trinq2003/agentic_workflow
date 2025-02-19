from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Self, Literal
from functools import lru_cache

from base_classes.memory.management_term import MemoryState, AccessPermission, MemoryType

@dataclass
class AbstractMemoryAtom(ABC):
    """Hierarchical memory unit combining architectural and agentic features"""
    _uuid: uuid.UUID # Globlally unique identifier
    _address: str # Relative address in the memory block
    _data: Any # Data stored in the memory atom
    _access_count: int
    _last_accessed: datetime
    _permission: Dict[str, AccessPermission] # Access permissions for different system components
    _state: MemoryState
    _identifying_features: Any

    _list_of_mematom_ids: List[uuid.UUID] = []
    _mematom_instances_by_id: Dict[str, Self] = {}
    def __init__(self, address: int, data: Any):
        self._uuid: uuid.UUID = uuid.uuid4()
        self._address: str = address
        self._data: Any = data
        self._access_count: int = 0
        self._last_accessed: datetime = datetime.now()
        self._permission: Dict[str, AccessPermission] = {}
        self._state: MemoryState = MemoryState.USED
        
        self._init_identifying_features()
        
        if self._uuid in self.__class__._mematom_instances_by_id.keys():
            raise ValueError(f"❌ Memory Atom ID {self._uuid} is already initiated.")
        else:
            self.__class__._mematom_instances_by_id[self._uuid] = self

    def _update_access(self):
        self._last_accessed = datetime.now()
        self._access_count += 1

    def read(self, requester: str):
        if self._check_permission(requester) > AccessPermission.NO_ACCESS:
            self._update_access()
            return self._data
        raise PermissionError(f"❌ {requester} lacks read permissions")

    def append_write(self, requester: Any, data: Any):
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
    
    def over_write(self, requester: Any, data: Any):
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
    
    @abstractmethod
    def _check_permission(self, requester: Any):
        """Return the access permission level for the requester for this memory instance.

        Args:
            requester (Any): The system component requesting access (agent, tool, operator, ...).
        """
        pass
    
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
        