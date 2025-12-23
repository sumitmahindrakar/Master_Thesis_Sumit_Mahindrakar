"""
Enhanced MDPA Mesh Refiner for Sensitivity Analysis
====================================================
Generates BOTH primary and dual MDPA files from template.

Features:
- Refines mesh with specified subdivisions
- Auto-generates dual MDPA with kink at response location
- Creates all necessary SubModelParts for dual analysis
- Returns hinge node IDs for use in dual analysis

Author: SA Pipeline
"""

import os
import sys
import shutil
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.config_loader import load_config, Config, create_directories
except ImportError:
    # Fallback for standalone testing
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SubModelPart:
    """Represents a SubModelPart in the MDPA file."""
    name: str
    nodes: List[int] = field(default_factory=list)
    elements: List[int] = field(default_factory=list)
    conditions: List[int] = field(default_factory=list)


@dataclass
class MdpaData:
    """Container for all MDPA file data."""
    header_lines: List[str] = field(default_factory=list)
    model_part_data: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    elements: Dict[int, dict] = field(default_factory=dict)
    element_type: str = ""
    elemental_data: Dict[str, Dict[int, str]] = field(default_factory=dict)
    conditions: Dict[int, dict] = field(default_factory=dict)
    condition_type: str = ""
    condition_num_nodes: int = 2
    sub_model_parts: Dict[str, SubModelPart] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of mesh refinement operation."""
    primary_mdpa_path: str
    dual_mdpa_path: str
    hinge_node_left: int
    hinge_node_right: int
    response_location: Tuple[float, float, float]
    n_nodes_primary: int
    n_nodes_dual: int
    n_elements: int


# =============================================================================
# MDPA PARSER
# =============================================================================

def detect_condition_nodes(condition_type: str) -> int:
    """Detect number of nodes from condition type name."""
    match = re.search(r'(\d)N', condition_type)
    if match:
        return int(match.group(1))
    return 2


def parse_mdpa(filename: str) -> MdpaData:
    """Parse an MDPA file and extract all data."""
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MDPA file not found: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = MdpaData()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # ModelPartData
        if line.startswith("Begin ModelPartData"):
            while i < len(lines) and not lines[i].strip().startswith("End ModelPartData"):
                data.model_part_data.append(lines[i])
                i += 1
            data.model_part_data.append(lines[i])
            i += 1
            continue

        # Properties
        if line.startswith("Begin Properties"):
            while i < len(lines) and not lines[i].strip().startswith("End Properties"):
                data.properties.append(lines[i])
                i += 1
            data.properties.append(lines[i])
            i += 1
            continue

        # Nodes
        if line.startswith("Begin Nodes"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End Nodes"):
                node_line = lines[i].strip()
                if node_line and not node_line.startswith("//"):
                    parts = node_line.split()
                    if len(parts) >= 4:
                        node_id = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        data.nodes[node_id] = (x, y, z)
                i += 1
            i += 1
            continue

        # Elements
        if line.startswith("Begin Elements"):
            parts = line.split("Begin Elements")
            if len(parts) > 1:
                data.element_type = parts[1].strip().split("//")[0].strip()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End Elements"):
                elem_line = lines[i].strip()
                if elem_line and not elem_line.startswith("//"):
                    parts = elem_line.split()
                    if len(parts) >= 4:
                        elem_id = int(parts[0])
                        prop_id = int(parts[1])
                        nodes = [int(p) for p in parts[2:]]
                        data.elements[elem_id] = {'property': prop_id, 'nodes': nodes}
                i += 1
            i += 1
            continue

        # ElementalData
        if line.startswith("Begin ElementalData"):
            data_name = line.split("Begin ElementalData")[1].strip().split("//")[0].strip()
            if data_name not in data.elemental_data:
                data.elemental_data[data_name] = {}
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End ElementalData"):
                elem_line = lines[i].strip()
                if elem_line and not elem_line.startswith("//"):
                    parts = elem_line.split(None, 1)
                    if len(parts) >= 2:
                        elem_id = int(parts[0])
                        value = parts[1]
                        data.elemental_data[data_name][elem_id] = value
                i += 1
            i += 1
            continue

        # Conditions
        if line.startswith("Begin Conditions"):
            parts = line.split("Begin Conditions")
            if len(parts) > 1:
                data.condition_type = parts[1].strip().split("//")[0].strip()
                data.condition_num_nodes = detect_condition_nodes(data.condition_type)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End Conditions"):
                cond_line = lines[i].strip()
                if cond_line and not cond_line.startswith("//"):
                    parts = cond_line.split()
                    if len(parts) >= 2 + data.condition_num_nodes:
                        cond_id = int(parts[0])
                        prop_id = int(parts[1])
                        nodes = [int(parts[j]) for j in range(2, 2 + data.condition_num_nodes)]
                        data.conditions[cond_id] = {'property': prop_id, 'nodes': nodes}
                i += 1
            i += 1
            continue

        # SubModelPart
        if line.startswith("Begin SubModelPart"):
            smp_name = line.split("Begin SubModelPart")[1].strip().split("//")[0].strip()
            smp = SubModelPart(name=smp_name)
            i += 1

            while i < len(lines) and not lines[i].strip().startswith("End SubModelPart"):
                smp_line = lines[i].strip()

                if smp_line.startswith("Begin SubModelPartNodes"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("End SubModelPartNodes"):
                        node_line = lines[i].strip()
                        if node_line and not node_line.startswith("//"):
                            smp.nodes.append(int(node_line))
                        i += 1
                    i += 1
                    continue

                if smp_line.startswith("Begin SubModelPartElements"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("End SubModelPartElements"):
                        elem_line = lines[i].strip()
                        if elem_line and not elem_line.startswith("//"):
                            smp.elements.append(int(elem_line))
                        i += 1
                    i += 1
                    continue

                if smp_line.startswith("Begin SubModelPartConditions"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("End SubModelPartConditions"):
                        cond_line = lines[i].strip()
                        if cond_line and not cond_line.startswith("//"):
                            smp.conditions.append(int(cond_line))
                        i += 1
                    i += 1
                    continue

                i += 1

            data.sub_model_parts[smp_name] = smp
            i += 1
            continue

        i += 1

    return data


# =============================================================================
# MESH REFINEMENT
# =============================================================================

def refine_mesh(data: MdpaData, subdivisions: int) -> MdpaData:
    """
    Refine mesh by subdividing elements.
    Returns refined mesh with clean sequential numbering.
    """
    
    original_node_ids = sorted(data.nodes.keys())
    original_elem_ids = sorted(data.elements.keys())
    
    old_to_new_node: Dict[int, int] = {}
    for new_id, old_id in enumerate(original_node_ids, start=1):
        old_to_new_node[old_id] = new_id
    
    refined = MdpaData()
    refined.header_lines = data.header_lines.copy()
    refined.model_part_data = data.model_part_data.copy()
    refined.properties = data.properties.copy()
    refined.element_type = data.element_type
    refined.condition_type = data.condition_type
    refined.condition_num_nodes = data.condition_num_nodes
    
    node_id_mapping: Dict[int, int] = {}
    intermediate_nodes_per_elem: Dict[int, List[int]] = {}
    
    next_new_node_id = 1
    seen_original_nodes = set()
    
    # Process elements in order
    for old_elem_id in original_elem_ids:
        elem = data.elements[old_elem_id]
        n1_old, n2_old = elem['nodes']
        
        if n1_old not in seen_original_nodes:
            seen_original_nodes.add(n1_old)
            node_id_mapping[n1_old] = next_new_node_id
            refined.nodes[next_new_node_id] = data.nodes[n1_old]
            next_new_node_id += 1
        
        p1 = data.nodes[n1_old]
        p2 = data.nodes[n2_old]
        
        new_intermediate_ids = []
        for j in range(1, subdivisions):
            t = j / subdivisions
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            z = p1[2] + t * (p2[2] - p1[2])
            refined.nodes[next_new_node_id] = (x, y, z)
            new_intermediate_ids.append(next_new_node_id)
            next_new_node_id += 1
        
        intermediate_nodes_per_elem[old_elem_id] = new_intermediate_ids
        
        if n2_old not in seen_original_nodes:
            seen_original_nodes.add(n2_old)
            node_id_mapping[n2_old] = next_new_node_id
            refined.nodes[next_new_node_id] = data.nodes[n2_old]
            next_new_node_id += 1
    
    # Create refined elements
    next_elem_id = 1
    old_to_new_elements: Dict[int, List[int]] = {}
    
    for old_elem_id in original_elem_ids:
        elem = data.elements[old_elem_id]
        n1_old, n2_old = elem['nodes']
        
        n1_new = node_id_mapping[n1_old]
        n2_new = node_id_mapping[n2_old]
        intermediate = intermediate_nodes_per_elem[old_elem_id]
        
        all_nodes = [n1_new] + intermediate + [n2_new]
        
        new_elem_ids = []
        for j in range(subdivisions):
            refined.elements[next_elem_id] = {
                'property': elem['property'],
                'nodes': [all_nodes[j], all_nodes[j + 1]]
            }
            new_elem_ids.append(next_elem_id)
            next_elem_id += 1
        
        old_to_new_elements[old_elem_id] = new_elem_ids
    
    # Handle elemental data
    for data_name, elem_data in data.elemental_data.items():
        refined.elemental_data[data_name] = {}
        for old_elem_id, value in elem_data.items():
            for new_elem_id in old_to_new_elements.get(old_elem_id, []):
                refined.elemental_data[data_name][new_elem_id] = value
    
    # Handle conditions
    next_cond_id = 1
    old_to_new_conditions: Dict[int, List[int]] = {}
    
    sorted_cond_ids = sorted(data.conditions.keys())
    
    for old_cond_id in sorted_cond_ids:
        cond = data.conditions[old_cond_id]
        
        if data.condition_num_nodes == 1:
            old_node = cond['nodes'][0]
            new_node = node_id_mapping[old_node]
            refined.conditions[next_cond_id] = {
                'property': cond['property'],
                'nodes': [new_node]
            }
            old_to_new_conditions[old_cond_id] = [next_cond_id]
            next_cond_id += 1
        else:
            n1_old, n2_old = cond['nodes']
            
            found_elem = None
            for old_elem_id in original_elem_ids:
                elem = data.elements[old_elem_id]
                if set(elem['nodes']) == {n1_old, n2_old}:
                    found_elem = old_elem_id
                    break
            
            if found_elem is not None:
                elem = data.elements[found_elem]
                n1_new = node_id_mapping[n1_old]
                n2_new = node_id_mapping[n2_old]
                intermediate = intermediate_nodes_per_elem[found_elem]
                
                if elem['nodes'][0] == n1_old:
                    all_nodes = [n1_new] + intermediate + [n2_new]
                else:
                    all_nodes = [n2_new] + intermediate + [n1_new]
                
                new_cond_ids = []
                for j in range(subdivisions):
                    refined.conditions[next_cond_id] = {
                        'property': cond['property'],
                        'nodes': [all_nodes[j], all_nodes[j + 1]]
                    }
                    new_cond_ids.append(next_cond_id)
                    next_cond_id += 1
                
                old_to_new_conditions[old_cond_id] = new_cond_ids
            else:
                new_nodes = [node_id_mapping[n] for n in cond['nodes']]
                refined.conditions[next_cond_id] = {
                    'property': cond['property'],
                    'nodes': new_nodes
                }
                old_to_new_conditions[old_cond_id] = [next_cond_id]
                next_cond_id += 1
    
    # Handle SubModelParts
    for smp_name, smp in data.sub_model_parts.items():
        new_smp = SubModelPart(name=smp_name)
        
        for old_node in smp.nodes:
            if old_node in node_id_mapping:
                new_smp.nodes.append(node_id_mapping[old_node])
        
        for old_elem_id in smp.elements:
            if old_elem_id in intermediate_nodes_per_elem:
                new_smp.nodes.extend(intermediate_nodes_per_elem[old_elem_id])
        
        new_smp.nodes = sorted(set(new_smp.nodes))
        
        for old_elem_id in smp.elements:
            if old_elem_id in old_to_new_elements:
                new_smp.elements.extend(old_to_new_elements[old_elem_id])
        new_smp.elements = sorted(new_smp.elements)
        
        for old_cond_id in smp.conditions:
            if old_cond_id in old_to_new_conditions:
                new_smp.conditions.extend(old_to_new_conditions[old_cond_id])
        new_smp.conditions = sorted(new_smp.conditions)
        
        refined.sub_model_parts[smp_name] = new_smp
    
    return refined


# =============================================================================
# KINK INSERTION FOR DUAL ANALYSIS
# =============================================================================

def find_nearest_node(nodes: Dict[int, Tuple[float, float, float]], 
                      target: Tuple[float, float, float],
                      tolerance: float = 1e-6) -> Tuple[int, Tuple[float, float, float], float]:
    """
    Find the node nearest to target coordinates.
    
    Returns
    -------
    Tuple of (node_id, coordinates, distance)
    """
    min_dist = float('inf')
    nearest_id = None
    nearest_coords = None
    
    for node_id, coords in nodes.items():
        dist = ((coords[0] - target[0])**2 + 
                (coords[1] - target[1])**2 + 
                (coords[2] - target[2])**2)**0.5
        
        if dist < min_dist:
            min_dist = dist
            nearest_id = node_id
            nearest_coords = coords
    
    return nearest_id, nearest_coords, min_dist


def find_elements_containing_node(elements: Dict[int, dict], 
                                   node_id: int) -> List[int]:
    """Find all elements that contain a given node."""
    result = []
    for elem_id, elem in elements.items():
        if node_id in elem['nodes']:
            result.append(elem_id)
    return sorted(result)


def detect_support_nodes(data: MdpaData) -> Dict[str, List[int]]:
    """
    Detect support nodes from SubModelParts.
    
    Returns dict with 'left' and 'right' support node lists.
    """
    supports = {'left': [], 'right': [], 'all': []}
    
    # Find nodes at min/max x coordinates
    if data.nodes:
        x_coords = [coords[0] for coords in data.nodes.values()]
        min_x = min(x_coords)
        max_x = max(x_coords)
        tolerance = (max_x - min_x) * 0.01 if max_x > min_x else 0.01
        
        for node_id, coords in data.nodes.items():
            if abs(coords[0] - min_x) < tolerance:
                supports['left'].append(node_id)
            if abs(coords[0] - max_x) < tolerance:
                supports['right'].append(node_id)
        
        supports['all'] = supports['left'] + supports['right']
    
    return supports


def create_dual_mdpa(refined_primary: MdpaData, 
                     response_location: Tuple[float, float, float]
                     ) -> Tuple[MdpaData, int, int]:
    """
    Create dual MDPA by inserting kink at response location.
    Nodes are renumbered to keep hinge nodes in sequential order.
    Support SubModelParts are preserved from the refined primary mesh.
    
    Parameters
    ----------
    refined_primary : MdpaData
        Refined primary mesh data
    response_location : tuple
        (x, y, z) coordinates for kink location
        
    Returns
    -------
    Tuple of (dual_mdpa_data, hinge_node_left, hinge_node_right)
    """
    
    # Find nearest node to response location
    kink_node_id, kink_coords, distance = find_nearest_node(
        refined_primary.nodes, response_location
    )
    
    print(f"\n  Kink insertion:")
    print(f"    Target location: ({response_location[0]:.4f}, {response_location[1]:.4f}, {response_location[2]:.4f})")
    print(f"    Nearest node: {kink_node_id} at ({kink_coords[0]:.4f}, {kink_coords[1]:.4f}, {kink_coords[2]:.4f})")
    print(f"    Distance: {distance:.6f}")
    
    # Find elements containing this node
    connected_elements = find_elements_containing_node(refined_primary.elements, kink_node_id)
    print(f"    Connected elements: {connected_elements}")
    
    if len(connected_elements) < 2:
        raise ValueError(
            f"Kink node {kink_node_id} is at boundary (only {len(connected_elements)} connected elements). "
            f"Choose an interior node for response location."
        )
    
    # Create dual mesh
    dual = MdpaData()
    dual.header_lines = refined_primary.header_lines.copy()
    dual.model_part_data = refined_primary.model_part_data.copy()
    dual.properties = refined_primary.properties.copy()
    dual.element_type = refined_primary.element_type
    
    # ===========================================================================
    # RENUMBER NODES: Insert duplicate hinge node right after original
    # ===========================================================================
    
    sorted_node_ids = sorted(refined_primary.nodes.keys())
    
    old_to_new_node: Dict[int, int] = {}
    new_node_id = 1
    
    hinge_left = None
    hinge_right = None
    
    for old_id in sorted_node_ids:
        old_to_new_node[old_id] = new_node_id
        
        if old_id == kink_node_id:
            hinge_left = new_node_id
            new_node_id += 1
            hinge_right = new_node_id
            new_node_id += 1
        else:
            new_node_id += 1
    
    # Build new nodes dict with proper ordering
    for old_id in sorted_node_ids:
        new_id = old_to_new_node[old_id]
        dual.nodes[new_id] = refined_primary.nodes[old_id]
        
        if old_id == kink_node_id:
            dual.nodes[hinge_right] = kink_coords
    
    print(f"    Hinge nodes: left={hinge_left}, right={hinge_right}")
    print(f"    Total nodes: {len(dual.nodes)} (was {len(refined_primary.nodes)})")
    
    # ===========================================================================
    # RENUMBER ELEMENTS: Update connectivity with new node IDs
    # ===========================================================================
    
    for elem_id, elem in refined_primary.elements.items():
        old_nodes = elem['nodes']
        new_nodes = []
        
        for i, old_node in enumerate(old_nodes):
            if old_node == kink_node_id:
                if i == 0:
                    new_nodes.append(hinge_right)
                else:
                    new_nodes.append(hinge_left)
            else:
                new_nodes.append(old_to_new_node[old_node])
        
        dual.elements[elem_id] = {
            'property': elem['property'],
            'nodes': new_nodes
        }
    
    # ===========================================================================
    # COPY ELEMENTAL DATA
    # ===========================================================================
    for data_name, elem_data in refined_primary.elemental_data.items():
        dual.elemental_data[data_name] = elem_data.copy()
    
    # ===========================================================================
    # CREATE CONDITIONS FOR DUAL ANALYSIS (PointMoment at hinge)
    # ===========================================================================
    dual.condition_type = "PointMomentCondition3D1N"
    dual.condition_num_nodes = 1
    dual.conditions = {
        1: {'property': 1, 'nodes': [hinge_left]}
    }
    
    # ===========================================================================
    # COPY AND UPDATE SUBMODELPARTS FROM PRIMARY
    # Preserve original support definitions, just update node IDs
    # ===========================================================================
    dual.sub_model_parts = {}
    
    # First, copy existing SubModelParts from primary (except load-related ones)
    for smp_name, smp in refined_primary.sub_model_parts.items():
        # Skip load conditions SubModelParts - we'll create DummyLoad instead
        if 'Load' in smp_name and 'Dummy' not in smp_name:
            continue
        
        new_smp = SubModelPart(name=smp_name)
        
        # Update node IDs
        for old_node in smp.nodes:
            if old_node in old_to_new_node:
                new_node = old_to_new_node[old_node]
                new_smp.nodes.append(new_node)
                
                # If this node is the kink node, we need to handle it specially
                # For support SubModelParts, we only keep the original node (hinge_left)
                # For Parts_Beam_Beams, we add both hinge nodes
                if old_node == kink_node_id and smp_name == 'Parts_Beam_Beams':
                    new_smp.nodes.append(hinge_right)
        
        new_smp.nodes = sorted(set(new_smp.nodes))
        
        # Update element IDs (elements don't change, just copy)
        new_smp.elements = smp.elements.copy()
        
        # Don't copy conditions from primary - dual has different conditions
        # new_smp.conditions = []
        
        dual.sub_model_parts[smp_name] = new_smp
    
    # ===========================================================================
    # ADD HINGE AND DUMMYLOAD SUBMODELPARTS
    # ===========================================================================
    
    # Hinge_Left
    dual.sub_model_parts['Hinge_Left'] = SubModelPart(
        name='Hinge_Left',
        nodes=[hinge_left]
    )
    
    # Hinge_Right
    dual.sub_model_parts['Hinge_Right'] = SubModelPart(
        name='Hinge_Right',
        nodes=[hinge_right]
    )
    
    # DummyLoad
    dual.sub_model_parts['DummyLoad'] = SubModelPart(
        name='DummyLoad',
        nodes=[hinge_left],
        conditions=[1]
    )
    
    return dual, hinge_left, hinge_right
# =============================================================================
# MDPA WRITER
# =============================================================================

def write_mdpa(data: MdpaData, filename: str) -> None:
    """Write MDPA data to file."""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        # ModelPartData
        if data.model_part_data:
            for line in data.model_part_data:
                f.write(line if line.endswith('\n') else line + '\n')
        else:
            f.write("Begin ModelPartData\n")
            f.write("End ModelPartData\n")
        f.write("\n")

        # Properties
        if data.properties:
            for line in data.properties:
                f.write(line if line.endswith('\n') else line + '\n')
        else:
            f.write("Begin Properties 1\n")
            f.write("End Properties\n")
        f.write("\n")

        # Nodes
        f.write("Begin Nodes\n")
        for node_id in sorted(data.nodes.keys()):
            x, y, z = data.nodes[node_id]
            f.write(f"    {node_id}   {x:.10f}   {y:.10f}   {z:.10f}\n")
        f.write("End Nodes\n\n")

        # Elements
        f.write(f"Begin Elements {data.element_type}\n")
        for elem_id in sorted(data.elements.keys()):
            elem = data.elements[elem_id]
            nodes_str = "   ".join(str(n) for n in elem['nodes'])
            f.write(f"    {elem_id}   {elem['property']}   {nodes_str}\n")
        f.write("End Elements\n\n")

        # ElementalData
        for data_name, elem_data in data.elemental_data.items():
            f.write(f"Begin ElementalData {data_name}\n")
            for elem_id in sorted(elem_data.keys()):
                f.write(f"    {elem_id} {elem_data[elem_id]}\n")
            f.write("End ElementalData\n\n")

        # Conditions
        if data.conditions:
            f.write(f"Begin Conditions {data.condition_type}\n")
            for cond_id in sorted(data.conditions.keys()):
                cond = data.conditions[cond_id]
                nodes_str = " ".join(str(n) for n in cond['nodes'])
                f.write(f"    {cond_id} {cond['property']} {nodes_str}\n")
            f.write("End Conditions\n\n")

        # SubModelParts
        for smp_name, smp in data.sub_model_parts.items():
            f.write(f"Begin SubModelPart {smp_name}\n")

            if smp.nodes:
                f.write("    Begin SubModelPartNodes\n")
                for node_id in sorted(smp.nodes):
                    f.write(f"        {node_id}\n")
                f.write("    End SubModelPartNodes\n")

            if smp.elements:
                f.write("    Begin SubModelPartElements\n")
                for elem_id in sorted(smp.elements):
                    f.write(f"        {elem_id}\n")
                f.write("    End SubModelPartElements\n")

            if smp.conditions:
                f.write("    Begin SubModelPartConditions\n")
                for cond_id in sorted(smp.conditions):
                    f.write(f"        {cond_id}\n")
                f.write("    End SubModelPartConditions\n")

            f.write("End SubModelPart\n\n")

    print(f"  Written: {filename}")


# =============================================================================
# JSON FILE UPDATER
# =============================================================================

def update_project_parameters(template_path: str, output_path: str,
                               mdpa_path: str, materials_path: str,
                               vtk_output_path: str) -> None:
    """
    Update ProjectParameters.json with correct paths.
    """
    
    with open(template_path, 'r') as f:
        params = json.load(f)
    
    # Get relative path without extension for mdpa
    mdpa_rel = os.path.splitext(mdpa_path)[0]
    
    # Update paths
    params['solver_settings']['model_import_settings']['input_filename'] = mdpa_rel
    params['solver_settings']['material_import_settings']['materials_filename'] = materials_path
    
    # Update VTK output path
    if 'output_processes' in params:
        if 'vtk_output' in params['output_processes']:
            for vtk_proc in params['output_processes']['vtk_output']:
                vtk_proc['Parameters']['output_path'] = vtk_output_path
        
        # Update GiD output path if present
        if 'gid_output' in params['output_processes']:
            for gid_proc in params['output_processes']['gid_output']:
                gid_proc['Parameters']['output_name'] = os.path.join(
                    os.path.dirname(vtk_output_path), 
                    'gid_output',
                    os.path.basename(mdpa_rel)
                )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"  Written: {output_path}")


def update_dual_project_parameters(template_path: str, output_path: str,
                                    mdpa_path: str, materials_path: str,
                                    vtk_output_path: str,
                                    hinge_left: int, hinge_right: int,
                                    supports: Dict[str, List[int]]) -> None:
    """
    Update ProjectParameters_dual.json with correct paths and hinge nodes.
    """
    
    with open(template_path, 'r') as f:
        params = json.load(f)
    
    mdpa_rel = os.path.splitext(mdpa_path)[0]
    
    # Update paths
    params['solver_settings']['model_import_settings']['input_filename'] = mdpa_rel
    params['solver_settings']['material_import_settings']['materials_filename'] = materials_path
    
    # Update VTK output path
    if 'output_processes' in params:
        if 'vtk_output' in params['output_processes']:
            for vtk_proc in params['output_processes']['vtk_output']:
                vtk_proc['Parameters']['output_path'] = vtk_output_path
    
    # Update constraint processes with correct SubModelPart names
    # The SubModelPart names in the JSON should match what we created in the MDPA
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"  Written: {output_path}")


# =============================================================================
# MAIN REFINEMENT FUNCTION
# =============================================================================

def refine_and_prepare(config: Config) -> RefinementResult:
    """
    Main function: Refine mesh and prepare all input files.
    
    Parameters
    ----------
    config : Config
        Configuration object from config.yaml
        
    Returns
    -------
    RefinementResult : Contains paths and hinge node IDs
    """
    
    print("\n" + "=" * 70)
    print("MESH REFINEMENT AND DUAL MDPA GENERATION")
    print("=" * 70)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories(config)
    
    # Parse template MDPA
    print(f"\n2. Parsing template: {config.paths.template_mdpa}")
    template_data = parse_mdpa(config.paths.template_mdpa)
    print(f"   Original: {len(template_data.nodes)} nodes, {len(template_data.elements)} elements")
    
    # Refine mesh for primary analysis
    print(f"\n3. Refining mesh with {config.mesh.subdivisions} subdivisions...")
    refined_primary = refine_mesh(template_data, config.mesh.subdivisions)
    print(f"   Refined: {len(refined_primary.nodes)} nodes, {len(refined_primary.elements)} elements")
    
    # Write primary MDPA
    print(f"\n4. Writing primary MDPA...")
    write_mdpa(refined_primary, config.paths.refined_mdpa_primary)
    
    # Create dual MDPA with kink
    print(f"\n5. Creating dual MDPA with kink...")
    response_loc = (config.response.x, config.response.y, 0.0)
    dual_data, hinge_left, hinge_right = create_dual_mdpa(refined_primary, response_loc)
    print(f"   Dual: {len(dual_data.nodes)} nodes, {len(dual_data.elements)} elements")
    
    # Write dual MDPA
    print(f"\n6. Writing dual MDPA...")
    write_mdpa(dual_data, config.paths.refined_mdpa_dual)
    
    # Copy materials file
    print(f"\n7. Copying materials file...")
    shutil.copy(config.paths.template_materials, config.paths.input_materials)
    print(f"  Copied: {config.paths.input_materials}")
    
    # Detect supports for dual analysis
    supports = detect_support_nodes(refined_primary)
    
    # Update ProjectParameters.json for primary
    print(f"\n8. Updating ProjectParameters.json (primary)...")
    update_project_parameters(
        config.paths.template_params_primary,
        config.paths.input_params_primary,
        config.paths.refined_mdpa_primary,
        config.paths.input_materials,
        os.path.dirname(config.paths.vtk_primary)
    )
    
    # Update ProjectParameters_dual.json
    print(f"\n9. Updating ProjectParameters_dual.json...")
    update_dual_project_parameters(
        config.paths.template_params_dual,
        config.paths.input_params_dual,
        config.paths.refined_mdpa_dual,
        config.paths.input_materials,
        os.path.dirname(config.paths.vtk_dual),
        hinge_left,
        hinge_right,
        supports
    )
    
    # Create result
    result = RefinementResult(
        primary_mdpa_path=config.paths.refined_mdpa_primary,
        dual_mdpa_path=config.paths.refined_mdpa_dual,
        hinge_node_left=hinge_left,
        hinge_node_right=hinge_right,
        response_location=response_loc,
        n_nodes_primary=len(refined_primary.nodes),
        n_nodes_dual=len(dual_data.nodes),
        n_elements=len(refined_primary.elements)
    )
    
    # Save hinge info for dual analysis
    hinge_info = {
        'hinge_node_left': hinge_left,
        'hinge_node_right': hinge_right,
        'response_location': list(response_loc)
    }
    hinge_info_path = os.path.join(config.paths.input_dir, 'hinge_info.json')
    with open(hinge_info_path, 'w') as f:
        json.dump(hinge_info, f, indent=4)
    print(f"\n  Saved hinge info: {hinge_info_path}")
    
    print("\n" + "=" * 70)
    print("REFINEMENT COMPLETE")
    print("=" * 70)
    print(f"\nPrimary MDPA: {result.primary_mdpa_path}")
    print(f"Dual MDPA:    {result.dual_mdpa_path}")
    print(f"Hinge nodes:  left={result.hinge_node_left}, right={result.hinge_node_right}")
    print("=" * 70)
    
    return result


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run refinement using config.yaml."""
    
    try:
        config = load_config()
        result = refine_and_prepare(config)
        
        print("\nRefinement successful!")
        print(f"  Primary nodes: {result.n_nodes_primary}")
        print(f"  Dual nodes: {result.n_nodes_dual}")
        print(f"  Elements: {result.n_elements}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)