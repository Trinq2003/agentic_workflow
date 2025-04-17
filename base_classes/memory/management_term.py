from enum import Enum

class MemoryState(Enum):
    FREE = 0 # Free to read and write data
    USED = 1 # Contains data and was used by at least a system component
    LOCKED = 2 # Locked, no system component can write data

class MemoryBlockState(Enum):
    EMPTY = 0 # Empty, no data in the memory block
    RAW_INPUT_ONLY = 1 # Raw input only, the system hasn't finished processing it yet
    RAW_INPUT_AND_OUTPUT = 2 # Raw input and output, the system processed the raw input and response raw data
    REFINED_INPUT = 3 # Refined input, the system processed the raw input and response refined data
    REFINED_INPUT_AND_OUTPUT = 4 # Refined input and output, the system processed the raw input and response refined data and outputed data
    FEATURE_ENGINEERED = 5 # Extracted features, the system processed the input and outputed data, and the data is refined and features are extracted