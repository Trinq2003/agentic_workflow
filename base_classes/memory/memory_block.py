import uuid
from typing import Dict, List, Self, Any
import textwrap

from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_features import MemoryBlockFeature
from base_classes.memory.management_term import MemoryBlockState
from base_classes.prompt import AbstractPrompt
from base_classes.system_component import SystemComponent
from base_classes.traceable_item import TimeTraceableItem
from base_classes.logger import HasLoggerClass

class AbstractMemoryBlock(TimeTraceableItem, HasLoggerClass):
    """
    The AbstractMemoryBlock class represents a collection of AbstractMemoryAtom instances.
    It serves as a container for storing chains of conversations or actions, providing a structured 
    way to manage and access memory atoms.

    Each memory block can contain multiple memory atoms, facilitating the organization of 
    related data and interactions over time.
    """
    _mem_block_id: uuid.UUID
    _memory_atoms: List[AbstractMemoryAtom] = []
    _mem_atom_graph: Dict[uuid.UUID, List[uuid.UUID]] = {} # Graph of memory atoms and their dependencies
    identifying_features: MemoryBlockFeature
    _input_query: str = "" # Input query from the user or system
    _output_response: str = "" # Output response from the system or assistant
    _refined_input_query: str = "" # Refined input query after processing
    # _refined_output_response: str = "" # Refined output response after processing
    _mem_block_state: MemoryBlockState
    _access_count: int = 0 # Access count for the memory block    
    _topic_container_ids: List[uuid.UUID] = []
    
    _memblock_instances_by_id: Dict[uuid.UUID, Self] = {}
    def __init__(self):
        TimeTraceableItem.__init__(self)
        HasLoggerClass.__init__(self)
        self._mem_block_id: uuid.UUID = uuid.uuid4()
        self._memory_atoms: List[AbstractMemoryAtom] = []
        self._mem_block_state: MemoryBlockState = MemoryBlockState.EMPTY
        
        # Initialize identifying_features with the proper structure
        self.identifying_features: MemoryBlockFeature = {
            "feature_for_raw_context": {
                "keywords": [],
                "input_embedding": None,
                "output_embedding": None,
                "context_embedding": None
            },
            "feature_for_refined_context": {
                "keywords": [],
                "refined_input_embedding": None
            }
        }
        
        if self._mem_block_id in self.__class__._memblock_instances_by_id.keys():
            self.logger.error(f"Memory Block ID {self._mem_block_id} is already initiated.")
            raise ValueError(f"❌ Memory Block ID {self._mem_block_id} is already initiated.")
        else:
            self.__class__._memblock_instances_by_id[self._mem_block_id] = self
    
    @classmethod
    def get_memblock_ids(cls) -> List[uuid.UUID]:
        """
        Get the list of memory block IDs.
        
        :return: The list of memory block IDs.
        :rtype: List[uuid.UUID]
        """
        return cls._memblock_instances_by_id.keys()
    @classmethod
    def get_memblock_instance_by_id(cls, mem_block_id: uuid.UUID) -> Self:
        """
        Retrieve an instance of the class by its ID.

        :param id: The unique identifier of the instance.
        :return: The instance if found, otherwise None.
        """
        return cls._memblock_instances_by_id.get(mem_block_id, None)
    
    @property
    def mem_block_id(self) -> uuid.UUID:
        return self._mem_block_id
    @property
    def memory_atoms(self) -> List[AbstractMemoryAtom]:
        return self._memory_atoms
    @property
    def mem_atom_graph(self) -> Dict[uuid.UUID, List[uuid.UUID]]:
        return self._mem_atom_graph
    @mem_atom_graph.setter
    def mem_atom_graph(self, graph: Dict[uuid.UUID, List[uuid.UUID]]) -> None:
        self._mem_atom_graph = graph
        self._sync_dependencies()

        #
        # --- Build and log a schematic representation of the dependency graph ---
        #
        def get_node_label(node_id):
            atom = AbstractMemoryAtom.get_mematom_instance_by_id(node_id)
            if not atom: return f"{str(node_id)[:4]}(?)"
            short_id = str(node_id)[:4]
            atom_type = atom._type.name[0] if atom._type else "?"
            return f"{short_id}({atom_type})"

        # 1. Build a forward-flow graph (conversation flow) from the dependency graph.
        #    In _mem_atom_graph, `A: [B]` means `A` depends on `B` (flow is B -> A).
        all_nodes = set(self._mem_atom_graph.keys())
        for deps in self._mem_atom_graph.values(): all_nodes.update(deps)
        
        flow_graph = {node: [] for node in all_nodes}
        for node, deps in self._mem_atom_graph.items():
            for dep in deps:
                flow_graph.setdefault(dep, []).append(node)

        # 2. Find all paths representing the flow of conversation.
        #    Start from nodes that don't depend on anything (flow starts).
        flow_starts = [node for node in all_nodes if not self._mem_atom_graph.get(node)]
        
        q = [[start] for start in flow_starts]
        paths = []
        while q:
            path = q.pop(0)
            last_node = path[-1]
            if not flow_graph.get(last_node): # It's an end-point of a flow
                paths.append(path)
                continue
            for neighbor in flow_graph[last_node]:
                q.append(path + [neighbor])
        
        if not paths:
            self.logger.debug("\n" + "="*50 + "\nNo memory paths found.\n" + "="*50)
            return

        # 3. Choose the longest path as the main line.
        paths.sort(key=len, reverse=True)
        main_path = paths[0]
        
        # 4. Draw the main path and record the position of each node.
        main_line_parts = [get_node_label(n) for n in main_path]
        main_line_str = "─".join(main_line_parts)
        
        node_positions = {}
        cursor = 0
        for i, node_id in enumerate(main_path):
            label_len = len(main_line_parts[i])
            node_positions[node_id] = {'mid': cursor + label_len // 2}
            cursor += label_len + 1 # +1 for the '─' separator

        # 5. Find all "bubbles" (alternative paths between two nodes on the main path).
        bubbles = []
        main_path_set = set(main_path)
        for i in range(len(main_path)):
            for j in range(i + 1, len(main_path)):
                u, v = main_path[i], main_path[j]
                
                # Find paths from u to v that don't touch the main path in between.
                path_q = [[u]]
                while path_q:
                    p = path_q.pop(0)
                    last = p[-1]
                    if last == v:
                        branch_nodes = p[1:-1]
                        # Ensure it's not the main path segment itself
                        is_main_segment = (main_path[i:j+1] == p)
                        if branch_nodes and not is_main_segment:
                            bubbles.append((u, v, tuple(branch_nodes)))
                        continue
                    
                    for neighbor in flow_graph.get(last, []):
                        if neighbor not in p and (neighbor == v or neighbor not in main_path_set):
                            path_q.append(p + [neighbor])
        
        # 6. Draw the bubbles on separate lines.
        canvas = {0: main_line_str}
        drawn_bubbles = set()
        for u, v, branch in sorted(list(set(bubbles)), key=lambda b: b[1]):
            if (u, v, branch) in drawn_bubbles: continue
            
            start_pos = node_positions[u]['mid']
            end_pos = node_positions[v]['mid']
            
            # Find an empty line level for this bubble
            level = 1
            while True:
                is_occupied = False
                if level in canvas:
                    # Very simple overlap check
                    if any(c != ' ' for c in canvas[level][start_pos:end_pos+1]):
                        is_occupied = True
                if not is_occupied: break
                level += 1

            line = list(canvas.get(level, ' ' * (cursor + 10)))
            while len(line) < end_pos + 1: line.append(' ')

            line[start_pos] = '└'
            line[end_pos] = '┘'
            
            branch_content = "─".join(get_node_label(n) for n in branch)
            bar_len = end_pos - start_pos - 1
            content_len = len(branch_content)
            
            for k in range(start_pos + 1, end_pos): line[k] = '─'
            
            if content_len <= bar_len:
                content_start = start_pos + 1 + (bar_len - content_len) // 2
                line[content_start : content_start+content_len] = list(branch_content)
            
            canvas[level] = "".join(line)
            drawn_bubbles.add((u, v, branch))
            
        # 7. Assemble final output string.
        output = "\nMemory Block's Atom Dependency Graph:\n" + "="*50 + "\n"
        for i in sorted(canvas.keys()):
            output += canvas[i] + "\n"
        output += "="*50
        self.logger.debug(output)
    @property
    def input_query(self) -> str:
        self._access_count += 1
        return self._input_query
    @input_query.setter
    def input_query(self, query: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.RAW_INPUT_ONLY:
            self.mem_block_state = MemoryBlockState.RAW_INPUT_ONLY
        self._input_query = query
    @property
    def output_response(self) -> str:
        self._access_count += 1
        return self._output_response
    @output_response.setter
    def output_response(self, response: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.INPUT_AND_OUTPUT:
            self.mem_block_state = MemoryBlockState.INPUT_AND_OUTPUT
        self._output_response = response
    @property
    def refined_input_query(self) -> str:
        self._access_count += 1
        return self._refined_input_query
    @refined_input_query.setter
    def refined_input_query(self, query: str) -> None:
        self._access_count += 1
        if self.mem_block_state < MemoryBlockState.REFINED_INPUT:
            self.mem_block_state = MemoryBlockState.REFINED_INPUT
        self._refined_input_query = query
    # @property
    # def refined_output_response(self) -> str:
    #     self._access_count += 1
    #     return self._refined_output_response
    # @refined_output_response.setter
    # def refined_output_response(self, response: str) -> None:
    #     self._access_count += 1
    #     if self.mem_block_state < MemoryBlockState.REFINED_INPUT_AND_OUTPUT:
    #         self.mem_block_state = MemoryBlockState.REFINED_INPUT_AND_OUTPUT
    #     self._refined_output_response = response
    @property
    def topic_container_ids(self) -> List[uuid.UUID]:
        return self._topic_container_ids
    @topic_container_ids.setter
    def topic_container_ids(self, topic_container_ids: List[uuid.UUID]) -> None:
        self._topic_container_ids = topic_container_ids
    @property
    def mem_block_state(self) -> MemoryBlockState:
        return self._mem_block_state
    @mem_block_state.setter
    def mem_block_state(self, state: MemoryBlockState) -> None:
        self._mem_block_state = state
    @property
    def access_count(self) -> int:
        return self._access_count
    
    def add_memory_atom(self, memory_atom: AbstractMemoryAtom) -> None:
        self._add_one_node_without_dependencies(memory_atom)
    
        for required_atom_id in memory_atom.required_atom:
            self._add_one_node_without_dependencies(AbstractMemoryAtom.get_mematom_instance_by_id(required_atom_id))
            if required_atom_id not in self._mem_atom_graph[memory_atom.mem_atom_id]:
                self._mem_atom_graph[memory_atom.mem_atom_id].append(required_atom_id)
                
        for requiring_atom_id in memory_atom.requiring_atom:
            self._add_one_node_without_dependencies(AbstractMemoryAtom.get_mematom_instance_by_id(requiring_atom_id))
            if requiring_atom_id not in self._mem_atom_graph[memory_atom.mem_atom_id]:
                self._mem_atom_graph[memory_atom.mem_atom_id].append(requiring_atom_id)
                
        self._sync_dependencies()
        
        # Extract input query from the first memory atom's prompt
        if len(self._memory_atoms) == 1:  # This is the first memory atom
            prompts = memory_atom.data.content.prompt
            for prompt in prompts:
                if prompt.get('role') == 'user':
                    # Extract the content from the user prompt
                    user_content = prompt.get('content', '')
                    if user_content:
                        self.input_query = user_content
                    break
        
        prompts = memory_atom.data.content.prompt
        roles = set()
        for prompt in prompts:
            roles.add(prompt.get('role'))
        
        if "user" in roles:
            if self.mem_block_state < MemoryBlockState.RAW_INPUT_ONLY:
                self.mem_block_state = MemoryBlockState.RAW_INPUT_ONLY
        if "assistant" in roles:
            if self.mem_block_state < MemoryBlockState.INPUT_AND_OUTPUT:
                self.mem_block_state = MemoryBlockState.INPUT_AND_OUTPUT
        
        self.logger.debug(f"Added memory atom {memory_atom.mem_atom_id} to memory block {self._mem_block_id}.")
    
    def _add_one_node_without_dependencies(self, memory_atom: AbstractMemoryAtom) -> None:
        if memory_atom.mem_atom_id in [ma_id.mem_atom_id for ma_id in self._memory_atoms]:
            self.logger.error(f"Memory Atom with ID {memory_atom.mem_atom_id} had already existed in Memory Block {self._mem_block_id}.")
            raise ValueError(f"❌ Memory Atom with ID {memory_atom.mem_atom_id} had already existed in Memory Block {self._mem_block_id}.")
        else:
            self._memory_atoms.append(AbstractMemoryAtom.get_mematom_instance_by_id(memory_atom.mem_atom_id))
            self._mem_atom_graph[memory_atom.mem_atom_id] = []
    
    def __str__(self):
        # TODO: Rewrite the __str__ method to be more informative and be specialized for keyword extraction task.
        memory_block_str = []
        for memory_atom in self._memory_atoms:
            memory_atom_str = str(memory_atom)
            memory_block_str.append(textwrap.indent(memory_atom_str, "\t"))
            
        prefix = f"MemoryBlock {self._mem_block_id}:\n"
        content = '\n'.join(memory_block_str)
        suffix = ""
        
        return prefix + content + suffix
                
    def get_memory_atom(self, requester: SystemComponent, mem_atom_id: uuid.UUID) -> AbstractMemoryAtom:
        pass
    
    def _search_similar_memory_atom(self, query: AbstractPrompt, top_k = 3) -> List[AbstractMemoryAtom]:
        """Search for similar memory atoms in the memory block from the input query.

        Args:
            query (AbstractPrompt): This can be a prompt of user, assistant, tool response, etc.
            top_k (int, optional): The maximum number of returned items. Defaults to 3.

        Returns:
            List[AbstractMemoryAtom]: A list of memory atoms that are similar to the query.
        """
        pass
    
    def _sync_dependencies(self) -> None:
        """
        Synchronize the dependencies of memory atoms in the memory block.
        """
        for mem_atom_id in self._mem_atom_graph.keys():
            memory_atom = AbstractMemoryAtom.get_mematom_instance_by_id(mem_atom_id)
            memory_atom.required_atom = self._mem_atom_graph[mem_atom_id]
            for required_atom_id in memory_atom.required_atom:
                required_atom = AbstractMemoryAtom.get_mematom_instance_by_id(required_atom_id)
                new_requiring_atom = required_atom.requiring_atom + [mem_atom_id]
                required_atom.requiring_atom = new_requiring_atom
    
    def __len__(self) -> int:
        return len(self._memory_atoms)