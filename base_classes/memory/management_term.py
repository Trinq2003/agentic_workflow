from enum import Enum

class MemoryState(Enum):
    FREE = 0 # Free to read and write data
    USED = 1 # Contains data and was used by at least a system component
    LOCKED = 2 # Locked, no system component can write data

class AccessPermission(Enum):
    NO_ACCESS = 0
    READ_ONLY = 1
    APPEND_ONLY = 2
    READ_WRITE = 3
    EXECUTE = 4
    OWNER = 5

class MemoryType(Enum):
    USER_INPUT = 1
    AGENT_THOUGHT = 2
    TOOL_OUTPUT = 3
    KNOWLEDGE = 4