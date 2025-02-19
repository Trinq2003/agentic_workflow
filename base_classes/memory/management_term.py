from enum import Enum

class MemoryState(Enum):
    FREE = 0
    USED = 1
    LOCKED = 2

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