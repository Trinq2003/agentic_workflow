from enum import IntEnum

class MemoryState(IntEnum):
    FREE = 0 # Free to read and write data
    USED = 1 # Contains data and was used by at least a system component
    LOCKED = 2 # Locked, no system component can write data
    
class MemoryAtomType(IntEnum):
    NONE_TYPE = 0 # No type, the system doesn't know what type it is
    USER_INPUT = 1 # User input, the system is waiting for user input
    ASSISTANT_OUTPUT = 2 # Assistant output, the system is waiting for assistant output
    TOOL_EXECUTION = 3 # Tool execution, the system is waiting for tool execution

class MemoryBlockState(IntEnum):
    EMPTY = 0 # Empty, no data in the memory block
    RAW_INPUT_ONLY = 1 # Input only, the system hasn't finished processing it yet
    REFINED_INPUT = 2 # Refined input, the system processed the raw input and response refined data
    INPUT_AND_OUTPUT = 3 # Input and output, the system processed the input and response raw data
    FEATURE_ENGINEERED = 4 # Extracted features, the system processed the input and outputed data, and the data is refined and features are extracted