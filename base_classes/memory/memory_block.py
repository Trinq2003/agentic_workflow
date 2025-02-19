from abc import ABC, abstractmethod
import uuid
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache
import time

from base_classes.memory.memory_atom import AbstractMemoryAtom

class AbstractMemoryBlock(ABC):
    _bank_id: uuid.UUID
    _memory_atoms: Dict[int, AbstractMemoryAtom]
    _access_matrix: Dict[str, Dict[str, Enum]]
    def __init__(self):
        self._bank_id: uuid.UUID = uuid.uuid4()