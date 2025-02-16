from typing import Dict, List, Set, Tuple
from abc import ABC

from base_classes.graph.edge import AbstractGraphEdge

class AbstractPath(ABC):
    _path_structure: Dict[Tuple[int, str], List[Tuple[int, str]]]

    def __init__(self, path_structure: Dict[Tuple[int, str], List[Tuple[int, str]]]) -> None:
        """
        Initializes an AbstractPath using a dictionary representation of paths.
        Sample input:
        {
            (1, 'obj_id_1'): [(2, 'obj_id_2'), (3, 'obj_id_3')],
            (2, 'obj_id_2'): [(4, 'obj_id_4')],
            (3, 'obj_id_3'): [(4, 'obj_id_4')],
            (4, 'obj_id_4'): []
        }

        :param path_structure: A dictionary where keys are node IDs and values are lists of adjacent node IDs.
        :param edge_class: Reference to AbstractGraphEdge to validate edges.
        """
        self._path_structure: Dict[Tuple[int, str], List[Tuple[int, str]]] = path_structure

    @property
    def path_structure(self) -> Dict[str, List[str]]:
        return self._path_structure
