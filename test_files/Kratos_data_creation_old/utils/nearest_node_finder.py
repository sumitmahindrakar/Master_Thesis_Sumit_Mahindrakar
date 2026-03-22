"""
Utility to find nearest node and element from coordinates.
"""

import math
from typing import Tuple, Dict, List, Optional


def parse_nodes_from_mdpa(mdpa_path: str) -> Dict[int, Tuple[float, float, float]]:
    """Parse nodes from MDPA file."""
    nodes = {}
    in_nodes_block = False
    
    with open(mdpa_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Begin Nodes"):
                in_nodes_block = True
                continue
            elif line.startswith("End Nodes"):
                break
            elif in_nodes_block and line and not line.startswith("//"):
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[node_id] = (x, y, z)
    
    return nodes


def parse_elements_from_mdpa(mdpa_path: str) -> Dict[int, List[int]]:
    """Parse elements from MDPA file."""
    elements = {}
    in_elements_block = False
    
    with open(mdpa_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Begin Elements"):
                in_elements_block = True
                continue
            elif line.startswith("End Elements"):
                break
            elif in_elements_block and line and not line.startswith("//"):
                parts = line.split()
                if len(parts) >= 4:
                    elem_id = int(parts[0])
                    # Skip property id (parts[1])
                    node_ids = [int(p) for p in parts[2:]]
                    elements[elem_id] = node_ids
    
    return elements


def find_nearest_node(
    target_coords: Tuple[float, float, float],
    nodes: Dict[int, Tuple[float, float, float]]
) -> Tuple[int, float]:
    """
    Find the nearest node to target coordinates.
    
    Returns:
        (node_id, distance)
    """
    min_dist = float('inf')
    nearest_node = None
    
    for node_id, (x, y, z) in nodes.items():
        dist = math.sqrt(
            (x - target_coords[0])**2 +
            (y - target_coords[1])**2 +
            (z - target_coords[2])**2
        )
        if dist < min_dist:
            min_dist = dist
            nearest_node = node_id
    
    return nearest_node, min_dist


def find_element_containing_node(
    node_id: int,
    elements: Dict[int, List[int]]
) -> Tuple[Optional[int], int]:
    """
    Find element containing the node and determine stress_location.
    
    Returns:
        (element_id, stress_location)
        stress_location: 0 if node is first in element, 1 if second
    """
    for elem_id, node_ids in elements.items():
        if node_id in node_ids:
            stress_location = node_ids.index(node_id)
            return elem_id, stress_location
    
    return None, 0


def find_nearest_element_and_location(
    target_coords: Tuple[float, float, float],
    mdpa_path: str
) -> Tuple[int, int, int, float]:
    """
    Find nearest node, its element, and stress location.
    
    Returns:
        (element_id, stress_location, nearest_node_id, distance)
    """
    nodes = parse_nodes_from_mdpa(mdpa_path)
    elements = parse_elements_from_mdpa(mdpa_path)
    
    nearest_node, distance = find_nearest_node(target_coords, nodes)
    element_id, stress_location = find_element_containing_node(nearest_node, elements)
    
    return element_id, stress_location, nearest_node, distance


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) >= 5:
        mdpa_file = sys.argv[1]
        x, y, z = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        
        elem_id, stress_loc, node_id, dist = find_nearest_element_and_location(
            (x, y, z), mdpa_file
        )
        print(f"Target: ({x}, {y}, {z})")
        print(f"Nearest node: {node_id} (distance: {dist:.4f})")
        print(f"Element: {elem_id}, stress_location: {stress_loc}")