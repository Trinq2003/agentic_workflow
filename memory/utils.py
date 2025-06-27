from typing import Dict, List
import uuid
from base_classes.memory.memory_atom import AbstractMemoryAtom


def visualize_dependency_graph(graph: Dict[uuid.UUID, List[uuid.UUID]]) -> str:
    """
    Visualize a dependency graph as a text-based schematic representation.
    
    :param graph: Dictionary where keys are node IDs and values are lists of dependent node IDs
    :type graph: Dict[uuid.UUID, List[uuid.UUID]]
    :return: String representation of the dependency graph
    :rtype: str
    """
    def get_node_label(node_id):
        atom = AbstractMemoryAtom.get_mematom_instance_by_id(node_id)
        if not atom: 
            return f"{str(node_id)[:4]}(?)"
        short_id = str(node_id)[:4]
        atom_type = atom._type.name[0] if atom._type else "?"
        return f"{short_id}({atom_type})"

    # 1. Build a forward-flow graph (conversation flow) from the dependency graph.
    #    In graph, `A: [B]` means `A` depends on `B` (flow is B -> A).
    all_nodes = set(graph.keys())
    for deps in graph.values(): 
        all_nodes.update(deps)
    
    flow_graph = {node: [] for node in all_nodes}
    for node, deps in graph.items():
        for dep in deps:
            flow_graph.setdefault(dep, []).append(node)

    # 2. Find all paths representing the flow of conversation.
    #    Start from nodes that don't depend on anything (flow starts).
    flow_starts = [node for node in all_nodes if not graph.get(node)]
    
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
    
    # 3. Find all nodes that are part of the graph but not in any path
    #    These are nodes that have dependencies but are not reachable from root nodes
    path_nodes = set()
    for path in paths:
        path_nodes.update(path)
    
    # Find nodes that are in the graph but not in any path
    unreachable_nodes = all_nodes - path_nodes
    
    # 4. Find isolated nodes (nodes with no connections at all)
    connected_nodes = path_nodes | unreachable_nodes
    isolated_nodes = all_nodes - connected_nodes
    
    # 5. Choose the longest path as the main line (if any paths exist).
    main_path = []
    if paths:
        paths.sort(key=len, reverse=True)
        main_path = paths[0]
    
    # 6. Draw the main path and record the position of each node.
    canvas = {}
    node_positions = {}
    cursor = 0
    
    if main_path:
        main_line_parts = [get_node_label(n) for n in main_path]
        main_line_str = "─".join(main_line_parts)
        canvas[0] = main_line_str
        
        for i, node_id in enumerate(main_path):
            label_len = len(main_line_parts[i])
            node_positions[node_id] = {'mid': cursor + label_len // 2}
            cursor += label_len + 1 # +1 for the '─' separator

        # 7. Find all "bubbles" (alternative paths between two nodes on the main path).
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
        
        # 8. Draw the bubbles on separate lines.
        canvas = {0: main_line_str}
        drawn_bubbles = set()
        drawn_nodes = set(main_path)
        
        for u, v, branch in sorted(list(set(bubbles)), key=lambda b: b[1]):
            if (u, v, branch) in drawn_bubbles: 
                continue
            
            start_pos = node_positions[u]['mid']
            end_pos = node_positions[v]['mid']
            
            level = 1
            while True:
                is_occupied = False
                if level in canvas and any(c != ' ' for c in canvas[level][start_pos:end_pos+1]):
                    is_occupied = True
                if not is_occupied: 
                    break
                level += 1

            line = list(canvas.get(level, ' ' * (cursor + 10)))
            while len(line) < end_pos + 1: 
                line.append(' ')

            line[start_pos] = '└'
            line[end_pos] = '┘'
            
            branch_content = "─".join(get_node_label(n) for n in branch)
            bar_len = end_pos - start_pos - 1
            content_len = len(branch_content)
            
            for k in range(start_pos + 1, end_pos): 
                line[k] = '─'
            
            if content_len <= bar_len:
                content_start = start_pos + 1 + (bar_len - content_len) // 2
                line[content_start : content_start+content_len] = list(branch_content)
            
            canvas[level] = "".join(line)
            drawn_bubbles.add((u, v, branch))
            drawn_nodes.update(branch)

        # 9. Draw dangling branches (diverge from main_path and don't return)
        for node_on_main in main_path:
            for neighbor in flow_graph.get(node_on_main, []):
                if neighbor not in drawn_nodes:
                    # This is the start of a dangling branch
                    branch_path = [neighbor]
                    drawn_nodes.add(neighbor)
                    curr = neighbor
                    # Follow the branch until it ends or hits an already drawn node
                    while flow_graph.get(curr):
                        next_node = flow_graph.get(curr)[0] # Assuming single path for dangling branches for simplicity
                        if next_node in drawn_nodes:
                            break
                        branch_path.append(next_node)
                        drawn_nodes.add(next_node)
                        curr = next_node
                    
                    # Draw the identified branch
                    start_pos = node_positions[node_on_main]['mid']
                    level = 1
                    while True:
                        if level not in canvas or canvas[level][start_pos] == ' ':
                            break
                        level += 1

                    line = list(canvas.get(level, ' ' * (cursor + 20)))
                    while len(line) < start_pos + 2: line.append(' ')
                    
                    line[start_pos] = '└'
                    branch_content = "─".join(get_node_label(n) for n in branch_path)
                    line[start_pos+1:start_pos+1+len(branch_content)] = list(branch_content)
                    canvas[level] = "".join(line)

    # 10. Add unreachable nodes to the visualization
    if unreachable_nodes:
        unreachable_level = max(canvas.keys()) + 1 if canvas else 0
        unreachable_labels = [get_node_label(node_id) for node_id in sorted(unreachable_nodes)]
        unreachable_line = "  ".join(unreachable_labels)
        canvas[unreachable_level] = f"Unreachable nodes: {unreachable_line}"
    
    # 11. Add isolated nodes to the visualization
    if isolated_nodes:
        isolated_level = max(canvas.keys()) + 1 if canvas else 0
        isolated_labels = [get_node_label(node_id) for node_id in sorted(isolated_nodes)]
        isolated_line = "  ".join(isolated_labels)
        canvas[isolated_level] = f"Isolated nodes: {isolated_line}"
    
    # 12. Assemble final output string.
    output = "Dependency Graph Visualization:\n" + "="*50 + "\n"
    
    if not canvas:
        output += "No nodes found in the graph.\n"
    else:
        for i in sorted(canvas.keys()):
            output += canvas[i] + "\n"
    
    # Add summary information
    output += "="*50 + "\n"
    output += f"Total nodes: {len(all_nodes)}\n"
    output += f"Connected nodes: {len(path_nodes)}\n"
    output += f"Unreachable nodes: {len(unreachable_nodes)}\n"
    output += f"Isolated nodes: {len(isolated_nodes)}\n"
    output += f"Paths found: {len(paths)}\n"
    
    return output
