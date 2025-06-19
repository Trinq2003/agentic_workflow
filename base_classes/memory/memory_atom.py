import uuid
from typing import List, Dict, Any, Self
import textwrap

from base_classes.memory.management_term import MemoryState, MemoryAtomType
from base_classes.traceable_item import TimeTraceableItem
from base_classes.logger import HasLoggerClass
from base_classes.memory.datatypes.data_item import PromptDataItem

class AbstractMemoryAtom(TimeTraceableItem, HasLoggerClass):
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
    _mem_atom_id: uuid.UUID # Globlally unique identifier
    _data: PromptDataItem # Data stored in the memory atom
    _access_count: int
    _state: MemoryState
    _type: MemoryAtomType # Type of the memory atom
    _required_atom: List[uuid.UUID] # List of memory atoms required for this atom to function
    _requiring_atom: List[uuid.UUID] # List of memory atoms requiring this atom to function

    _mematom_instances_by_id: Dict[uuid.UUID, Self] = {}
    def __init__(self, data: PromptDataItem, required_atom: List[uuid.UUID] = [], requiring_atom: List[uuid.UUID] = []):
        TimeTraceableItem.__init__(self)
        HasLoggerClass.__init__(self)
        self._mem_atom_id: uuid.UUID = uuid.uuid4()
        self._data: PromptDataItem = data
        self._required_atom: List[uuid.UUID] = required_atom
        self._requiring_atom: List[uuid.UUID] = requiring_atom
        self._access_count: int = 0
        self._state: MemoryState = MemoryState.USED
        prompt_role = self._data.content.prompt[0]['role']
        if prompt_role == "user":
            self._type: MemoryAtomType = MemoryAtomType.USER_INPUT
        elif prompt_role == "assistant":
            self._type: MemoryAtomType = MemoryAtomType.ASSISTANT_OUTPUT
        elif prompt_role == "tool":
            self._type: MemoryAtomType = MemoryAtomType.TOOL_EXECUTION
        
        if self._mem_atom_id in self.__class__._mematom_instances_by_id.keys():
            self.logger.error(f"Memory Atom ID {self._mem_atom_id} is already initiated.")
            raise ValueError(f"❌ Memory Atom ID {self._mem_atom_id} is already initiated.")
        else:
            self.__class__._mematom_instances_by_id[self._mem_atom_id] = self
        
        self.logger.debug(f"Memory Atom ID: {self._mem_atom_id} | Memory Atom type: {self._type}; Memory Atom state: {self._state}")
            
    @classmethod
    def get_mematom_ids(cls) -> List[uuid.UUID]:
        """
        Get the list of memory atom instances.

        :return: The list of memory atom instances.
        :rtype: List[uuid.UUID, Self]
        """
        return cls._mematom_instances_by_id.keys()
    @classmethod
    def get_mematom_instance_by_id(cls, mem_atom_id: uuid.UUID) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._mematom_instances_by_id.get(mem_atom_id, None)

    @property
    def mem_atom_id(self):
        return self._mem_atom_id
    @property
    def access_count(self):
        return self._access_count
    @property
    def state(self):
        return self._state
    @property
    def required_atom(self):
        return self._required_atom
    @property
    def data(self):
        return self._data
    @required_atom.setter
    def required_atom(self, required_atom: List[uuid.UUID]):
        self._required_atom = required_atom
    @property
    def requiring_atom(self):
        return self._requiring_atom
    @requiring_atom.setter
    def requiring_atom(self, requiring_atom: List[uuid.UUID]):
        self._requiring_atom = requiring_atom
        
    def __str__(self):
        # TODO: Change the wording method
        data_str = textwrap.indent(str(self._data),"\t")

        prefix = f"MemoryAtom {self._mem_atom_id}:\n"
        suffix = ""
        
        return prefix + data_str + suffix