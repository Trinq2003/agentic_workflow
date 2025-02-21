from enum import Enum

class MemoryState(Enum):
    FREE = 0 # Free to read and write data
    USED = 1 # Contains data and was used by at least a system component
    LOCKED = 2 # Locked, no system component can write data

class MemoryType(Enum):
    LOWER_READ_HIGHER_READ = 1
    LOWER_WRITE_HIGHER_READ = 2
    LOWER_READ_HIGHER_NO_ACCESS = 3
    LOWER_WRITE_HIGHER_NO_ACCESS = 4
    LOWER_NO_ACCESS_HIGHER_WRITE = 5
    LOWER_READ_HIGHER_WRITE = 6
    LOWER_WRITE_HIGHER_WRITE = 7