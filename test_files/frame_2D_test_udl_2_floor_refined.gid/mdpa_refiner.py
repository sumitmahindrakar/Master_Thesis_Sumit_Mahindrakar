#!/usr/bin/env python3
"""
MDPA Mesh Refinement Tool
Refines beam element meshes by subdividing elements into smaller segments.
"""

import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import copy


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float


@dataclass
class Element:
    id: int
    property_id: int
    node_ids: List[int]
    element_type: str = ""


@dataclass
class Condition:
    id: int
    property_id: int
    node_ids: List[int]
    condition_type: str = ""


@dataclass
class ElementalData:
    id: int
    values: List[float]


@dataclass
class SubModelPart:
    name: str
    nodes: List[int] = field(default_factory=list)
    elements: List[int] = field(default_factory=list)
    conditions: List[int] = field(default_factory=list)


@dataclass
class MDPAData:
    model_part_data: str = ""
    properties: Dict[int, str] = field(default_factory=dict)
    nodes: Dict[int, Node] = field(default_factory=dict)
    elements: Dict[int, Element] = field(default_factory=dict)
    element_type: str = ""
    element_comment: str = ""
    elemental_data: Dict[int, ElementalData] = field(default_factory=dict)
    elemental_data_name: str = ""
    elemental_data_comment: str = ""
    conditions: Dict[int, Condition] = field(default_factory=dict)
    condition_type: str = ""
    condition_comment: str = ""
    sub_model_parts: Dict[str, SubModelPart] = field(default_factory=dict)


def parse_mdpa(filepath: str) -> MDPAData:
    """Parse an MDPA file and extract all data."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = MDPAData()
    
    # Parse ModelPartData
    match = re.search(r'Begin ModelPartData\s*(.*?)\s*End ModelPartData', content, re.DOTALL)
    if match:
        data.model_part_data = match.group(1).strip()
    
    # Parse Properties
    for match in re.finditer(r'Begin Properties (\d+)\s*(.*?)\s*End Properties', content, re.DOTALL):
        prop_id = int(match.group(1))
        data.properties[prop_id] = match.group(2).strip()
    
    # Parse Nodes
    match = re.search(r'Begin Nodes\s*(.*?)\s*End Nodes', content, re.DOTALL)
    if match:
        nodes_text = match.group(1)
        for line in nodes_text.strip().split('\n'):
            line = line.split('//')[0].strip()  # Remove comments
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    data.nodes[node_id] = Node(node_id, x, y, z)
    
    # Parse Elements
    match = re.search(r'Begin Elements\s+(\w+)\s*(//.*?)?\n(.*?)\s*End Elements', content, re.DOTALL)
    if match:
        data.element_type = match.group(1)
        data.element_comment = match.group(2) if match.group(2) else ""
        elements_text = match.group(3)
        for line in elements_text.strip().split('\n'):
            line = line.split('//')[0].strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    elem_id = int(parts[0])
                    prop_id = int(parts[1])
                    node_ids = [int(p) for p in parts[2:]]
                    data.elements[elem_id] = Element(elem_id, prop_id, node_ids, data.element_type)
    
    # Parse ElementalData
    match = re.search(r'Begin ElementalData\s+(\w+)\s*(//.*?)?\n(.*?)\s*End ElementalData', content, re.DOTALL)
    if match:
        data.elemental_data_name = match.group(1)
        data.elemental_data_comment = match.group(2) if match.group(2) else ""
        elem_data_text = match.group(3)
        for line in elem_data_text.strip().split('\n'):
            line = line.split('//')[0].strip()
            if line:
                # Parse format: id [3]( x, y, z)
                match_data = re.match(r'(\d+)\s*\[3\]\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', line)
                if match_data:
                    elem_id = int(match_data.group(1))
                    values = [float(match_data.group(2)), float(match_data.group(3)), float(match_data.group(4))]
                    data.elemental_data[elem_id] = ElementalData(elem_id, values)
    
    # Parse Conditions
    match = re.search(r'Begin Conditions\s+(\w+)\s*(//.*?)?\n(.*?)\s*End Conditions', content, re.DOTALL)
    if match:
        data.condition_type = match.group(1)
        data.condition_comment = match.group(2) if match.group(2) else ""
        conditions_text = match.group(3)
        for line in conditions_text.strip().split('\n'):
            line = line.split('//')[0].strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    cond_id = int(parts[0])
                    prop_id = int(parts[1])
                    node_ids = [int(p) for p in parts[2:]]
                    data.conditions[cond_id] = Condition(cond_id, prop_id, node_ids, data.condition_type)
    
    # Parse SubModelParts
    pattern = r'Begin SubModelPart\s+(\S+)\s*(//.*?)?\n(.*?)\nEnd SubModelPart'
    for match in re.finditer(pattern, content, re.DOTALL):
        name = match.group(1)
        smp_content = match.group(3)
        
        smp = SubModelPart(name)
        
        # Parse nodes in submodelpart
        nodes_match = re.search(r'Begin SubModelPartNodes\s*(.*?)\s*End SubModelPartNodes', smp_content, re.DOTALL)
        if nodes_match:
            for line in nodes_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    smp.nodes.extend([int(n) for n in line.split()])
        
        # Parse elements in submodelpart
        elem_match = re.search(r'Begin SubModelPartElements\s*(.*?)\s*End SubModelPartElements', smp_content, re.DOTALL)
        if elem_match:
            for line in elem_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    smp.elements.extend([int(e) for e in line.split()])
        
        # Parse conditions in submodelpart
        cond_match = re.search(r'Begin SubModelPartConditions\s*(.*?)\s*End SubModelPartConditions', smp_content, re.DOTALL)
        if cond_match:
            for line in cond_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    smp.conditions.extend([int(c) for c in line.split()])
        
        data.sub_model_parts[name] = smp
    
    return data


def interpolate_node(node1: Node, node2: Node, fraction: float, new_id: int) -> Node:
    """Create an interpolated node between two nodes."""
    x = node1.x + fraction * (node2.x - node1.x)
    y = node1.y + fraction * (node2.y - node1.y)
    z = node1.z + fraction * (node2.z - node1.z)
    return Node(new_id, x, y, z)


def refine_mesh(data: MDPAData, subdivisions: int) -> MDPAData:
    """
    Refine the mesh by subdividing each element into 'subdivisions' parts.
    
    Args:
        data: Original MDPA data
        subdivisions: Number of subdivisions per original element
    
    Returns:
        Refined MDPA data
    """
    
    if subdivisions < 1:
        raise ValueError("Subdivisions must be at least 1")
    
    if subdivisions == 1:
        return copy.deepcopy(data)
    
    refined = MDPAData()
    refined.model_part_data = data.model_part_data
    refined.properties = copy.deepcopy(data.properties)
    refined.element_type = data.element_type
    refined.element_comment = data.element_comment
    refined.elemental_data_name = data.elemental_data_name
    refined.elemental_data_comment = data.elemental_data_comment
    refined.condition_type = data.condition_type
    refined.condition_comment = data.condition_comment
    
    # Copy original nodes
    refined.nodes = copy.deepcopy(data.nodes)
    
    # Track mappings for submodelparts
    element_mapping: Dict[int, List[int]] = {}  # old_elem_id -> [new_elem_ids]
    condition_mapping: Dict[int, List[int]] = {}  # old_cond_id -> [new_cond_ids]
    new_nodes_for_elements: Dict[int, List[int]] = {}  # old_elem_id -> [new_node_ids including endpoints]
    
    # Calculate next available IDs
    next_node_id = max(data.nodes.keys()) + 1 if data.nodes else 1
    next_element_id = 1
    next_condition_id = 1
    
    # Refine elements
    for old_elem_id, elem in sorted(data.elements.items()):
        if len(elem.node_ids) != 2:
            raise ValueError(f"Element {old_elem_id} has {len(elem.node_ids)} nodes, expected 2 for beam elements")
        
        node1_id, node2_id = elem.node_ids
        node1 = data.nodes[node1_id]
        node2 = data.nodes[node2_id]
        
        # Create intermediate nodes
        intermediate_node_ids = [node1_id]
        for i in range(1, subdivisions):
            fraction = i / subdivisions
            new_node = interpolate_node(node1, node2, fraction, next_node_id)
            refined.nodes[next_node_id] = new_node
            intermediate_node_ids.append(next_node_id)
            next_node_id += 1
        intermediate_node_ids.append(node2_id)
        
        new_nodes_for_elements[old_elem_id] = intermediate_node_ids
        
        # Create new elements
        new_elem_ids = []
        for i in range(subdivisions):
            new_elem = Element(
                next_element_id,
                elem.property_id,
                [intermediate_node_ids[i], intermediate_node_ids[i + 1]],
                elem.element_type
            )
            refined.elements[next_element_id] = new_elem
            
            # Copy elemental data (same orientation for all subdivided elements)
            if old_elem_id in data.elemental_data:
                refined.elemental_data[next_element_id] = ElementalData(
                    next_element_id,
                    copy.copy(data.elemental_data[old_elem_id].values)
                )
            
            new_elem_ids.append(next_element_id)
            next_element_id += 1
        
        element_mapping[old_elem_id] = new_elem_ids
    
    # Refine conditions (similar to elements)
    condition_new_nodes: Dict[int, List[int]] = {}
    
    for old_cond_id, cond in sorted(data.conditions.items()):
        if len(cond.node_ids) != 2:
            raise ValueError(f"Condition {old_cond_id} has {len(cond.node_ids)} nodes, expected 2")
        
        node1_id, node2_id = cond.node_ids
        node1 = data.nodes[node1_id]
        node2 = data.nodes[node2_id]
        
        # Check if this edge was already refined as an element
        edge_key = tuple(sorted([node1_id, node2_id]))
        existing_nodes = None
        
        for old_elem_id, node_list in new_nodes_for_elements.items():
            elem_edge = tuple(sorted(data.elements[old_elem_id].node_ids))
            if edge_key == elem_edge:
                existing_nodes = node_list if node_list[0] == node1_id else node_list[::-1]
                break
        
        if existing_nodes:
            intermediate_node_ids = existing_nodes
        else:
            # Create intermediate nodes for condition
            intermediate_node_ids = [node1_id]
            for i in range(1, subdivisions):
                fraction = i / subdivisions
                new_node = interpolate_node(node1, node2, fraction, next_node_id)
                refined.nodes[next_node_id] = new_node
                intermediate_node_ids.append(next_node_id)
                next_node_id += 1
            intermediate_node_ids.append(node2_id)
        
        condition_new_nodes[old_cond_id] = intermediate_node_ids
        
        # Create new conditions
        new_cond_ids = []
        for i in range(subdivisions):
            new_cond = Condition(
                next_condition_id,
                cond.property_id,
                [intermediate_node_ids[i], intermediate_node_ids[i + 1]],
                cond.condition_type
            )
            refined.conditions[next_condition_id] = new_cond
            new_cond_ids.append(next_condition_id)
            next_condition_id += 1
        
        condition_mapping[old_cond_id] = new_cond_ids
    
    # Update SubModelParts
    for name, smp in data.sub_model_parts.items():
        new_smp = SubModelPart(name)
        
        # Collect all nodes (original + new intermediate nodes)
        new_nodes_set = set()
        for node_id in smp.nodes:
            new_nodes_set.add(node_id)
        
        # Add intermediate nodes from elements in this submodelpart
        for elem_id in smp.elements:
            if elem_id in new_nodes_for_elements:
                new_nodes_set.update(new_nodes_for_elements[elem_id])
        
        new_smp.nodes = sorted(list(new_nodes_set))
        
        # Map old elements to new elements
        for elem_id in smp.elements:
            if elem_id in element_mapping:
                new_smp.elements.extend(element_mapping[elem_id])
        new_smp.elements = sorted(new_smp.elements)
        
        # Map old conditions to new conditions
        for cond_id in smp.conditions:
            if cond_id in condition_mapping:
                new_smp.conditions.extend(condition_mapping[cond_id])
        new_smp.conditions = sorted(new_smp.conditions)
        
        refined.sub_model_parts[name] = new_smp
    
    return refined


def write_mdpa(data: MDPAData, filepath: str):
    """Write MDPA data to a file."""
    
    with open(filepath, 'w') as f:
        # ModelPartData
        f.write("Begin ModelPartData\n")
        if data.model_part_data:
            f.write(f"{data.model_part_data}\n")
        f.write("End ModelPartData\n\n")
        
        # Properties
        for prop_id, prop_content in sorted(data.properties.items()):
            f.write(f"Begin Properties {prop_id}\n")
            if prop_content:
                f.write(f"{prop_content}\n")
            f.write("End Properties\n\n")
        
        # Nodes
        f.write("\nBegin Nodes\n")
        for node_id in sorted(data.nodes.keys()):
            node = data.nodes[node_id]
            f.write(f"    {node.id}   {node.x:.10f}   {node.y:.10f}   {node.z:.10f}\n")
        f.write("End Nodes\n\n")
        
        # Elements
        if data.elements:
            comment = f"{data.element_comment}" if data.element_comment else ""
            f.write(f"\nBegin Elements {data.element_type}{comment}\n")
            for elem_id in sorted(data.elements.keys()):
                elem = data.elements[elem_id]
                nodes_str = "     ".join(str(n) for n in elem.node_ids)
                f.write(f"        {elem.id}          {elem.property_id}     {nodes_str} \n")
            f.write("End Elements\n")
        
        # ElementalData
        if data.elemental_data:
            comment = f" {data.elemental_data_comment}" if data.elemental_data_comment else ""
            f.write(f"\nBegin ElementalData {data.elemental_data_name}{comment}\n")
            for elem_id in sorted(data.elemental_data.keys()):
                ed = data.elemental_data[elem_id]
                f.write(f"    {ed.id} [3]( {ed.values[0]:.10f},   {ed.values[1]:.10f},  {ed.values[2]:.10f})\n")
            f.write("End ElementalData\n")
        
        # Conditions
        if data.conditions:
            comment = f"{data.condition_comment}" if data.condition_comment else ""
            f.write(f"\nBegin Conditions {data.condition_type}{comment}\n")
            for cond_id in sorted(data.conditions.keys()):
                cond = data.conditions[cond_id]
                nodes_str = " ".join(str(n) for n in cond.node_ids)
                f.write(f"    {cond.id} {cond.property_id} {nodes_str}\n")
            f.write("End Conditions\n")
        
        # SubModelParts
        for name in sorted(data.sub_model_parts.keys()):
            smp = data.sub_model_parts[name]
            f.write(f"\nBegin SubModelPart {name}\n")
            
            if smp.nodes:
                f.write("    Begin SubModelPartNodes\n")
                for node_id in smp.nodes:
                    f.write(f"            {node_id}\n")
                f.write("    End SubModelPartNodes\n")
            
            if smp.elements:
                f.write("    Begin SubModelPartElements\n")
                for elem_id in smp.elements:
                    f.write(f"            {elem_id}\n")
                f.write("    End SubModelPartElements\n")
            
            if smp.conditions:
                f.write("    Begin SubModelPartConditions\n")
                for cond_id in smp.conditions:
                    f.write(f"            {cond_id}\n")
                f.write("    End SubModelPartConditions\n")
            
            f.write("End SubModelPart\n")


def main():
    parser = argparse.ArgumentParser(
        description='Refine MDPA mesh files by subdividing elements.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Subdivide each element into 3 parts
    python mdpa_refiner.py input.mdpa -s 3 -o output.mdpa
    
    # Subdivide each element into 5 parts with verbose output
    python mdpa_refiner.py model.mdpa -s 5 -o refined_model.mdpa -v
        """
    )
    
    parser.add_argument('input', help='Input MDPA file')
    parser.add_argument('-o', '--output', required=True, help='Output MDPA file')
    parser.add_argument('-s', '--subdivisions', type=int, default=2,
                        help='Number of subdivisions per element (default: 2)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Reading input file: {args.input}")
    
    # Parse input file
    data = parse_mdpa(args.input)
    
    if args.verbose:
        print(f"Original mesh:")
        print(f"  - Nodes: {len(data.nodes)}")
        print(f"  - Elements: {len(data.elements)}")
        print(f"  - Conditions: {len(data.conditions)}")
        print(f"  - SubModelParts: {len(data.sub_model_parts)}")
    
    # Refine mesh
    refined = refine_mesh(data, args.subdivisions)
    
    if args.verbose:
        print(f"\nRefined mesh (subdivisions={args.subdivisions}):")
        print(f"  - Nodes: {len(refined.nodes)}")
        print(f"  - Elements: {len(refined.elements)}")
        print(f"  - Conditions: {len(refined.conditions)}")
        print(f"  - SubModelParts: {len(refined.sub_model_parts)}")
    
    # Write output file
    write_mdpa(refined, args.output)
    
    if args.verbose:
        print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()