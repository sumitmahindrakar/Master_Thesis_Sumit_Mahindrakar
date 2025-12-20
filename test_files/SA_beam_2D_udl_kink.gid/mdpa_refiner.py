"""
MDPA Mesh Refiner - Handles both 1-node and 2-node conditions
Preserves node identity for coincident nodes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import re


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


def detect_condition_nodes(condition_type: str) -> int:
    """Detect number of nodes from condition type name."""
    match = re.search(r'(\d)N', condition_type)
    if match:
        return int(match.group(1))
    return 2


def parse_mdpa(filename: str) -> MdpaData:
    """Parse an MDPA file and extract all data."""
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


def refine_mesh(data: MdpaData, subdivisions: int) -> MdpaData:
    """
    Refine mesh by subdividing elements.
    Returns refined mesh with clean sequential numbering.
    """
    # Build ordered list of original nodes (preserving their order/identity)
    original_node_ids = sorted(data.nodes.keys())
    
    # Build ordered list of original elements
    original_elem_ids = sorted(data.elements.keys())
    
    # Create mapping from old node ID to new node ID
    # Original nodes keep relative order but get renumbered 1, 2, 3...
    old_to_new_node: Dict[int, int] = {}
    for new_id, old_id in enumerate(original_node_ids, start=1):
        old_to_new_node[old_id] = new_id
    
    # Prepare refined data
    refined = MdpaData()
    refined.header_lines = data.header_lines.copy()
    refined.model_part_data = data.model_part_data.copy()
    refined.properties = data.properties.copy()
    refined.element_type = data.element_type
    refined.condition_type = data.condition_type
    refined.condition_num_nodes = data.condition_num_nodes
    
    # We'll build nodes in a specific order:
    # For each element (in order), we add: first_node, intermediate_nodes..., (last_node handled by next element or at end)
    # This ensures nodes along the mesh are sequential
    
    # First pass: determine node ordering based on element traversal
    # We want nodes to be numbered in the order they appear along elements
    
    # Track which original nodes we've seen and new intermediate nodes
    node_order: List[Tuple[int, Tuple[float, float, float]]] = []  # (old_id or -1 for new, coords)
    node_id_mapping: Dict[int, int] = {}  # old_node_id -> final_new_id
    intermediate_nodes_per_elem: Dict[int, List[int]] = {}  # old_elem_id -> list of new node final IDs
    
    next_new_node_id = 1
    seen_original_nodes = set()
    
    # Process elements in order to build node sequence
    for old_elem_id in original_elem_ids:
        elem = data.elements[old_elem_id]
        n1_old, n2_old = elem['nodes']
        
        # Add first node if not seen
        if n1_old not in seen_original_nodes:
            seen_original_nodes.add(n1_old)
            node_id_mapping[n1_old] = next_new_node_id
            refined.nodes[next_new_node_id] = data.nodes[n1_old]
            next_new_node_id += 1
        
        # Add intermediate nodes
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
        
        # Add second node if not seen
        if n2_old not in seen_original_nodes:
            seen_original_nodes.add(n2_old)
            node_id_mapping[n2_old] = next_new_node_id
            refined.nodes[next_new_node_id] = data.nodes[n2_old]
            next_new_node_id += 1
    
    # Now create refined elements
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
    
    # Sort conditions by their first node for consistent output
    sorted_cond_ids = sorted(data.conditions.keys())
    
    for old_cond_id in sorted_cond_ids:
        cond = data.conditions[old_cond_id]
        
        if data.condition_num_nodes == 1:
            # Point condition - just remap the node
            old_node = cond['nodes'][0]
            new_node = node_id_mapping[old_node]
            refined.conditions[next_cond_id] = {
                'property': cond['property'],
                'nodes': [new_node]
            }
            old_to_new_conditions[old_cond_id] = [next_cond_id]
            next_cond_id += 1
        else:
            # Line condition - find the element and subdivide
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
                
                # Maintain correct direction
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
                # Element not found, just remap nodes
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
        
        # Map original nodes
        for old_node in smp.nodes:
            if old_node in node_id_mapping:
                new_smp.nodes.append(node_id_mapping[old_node])
        
        # Add intermediate nodes from elements in this SMP
        for old_elem_id in smp.elements:
            if old_elem_id in intermediate_nodes_per_elem:
                new_smp.nodes.extend(intermediate_nodes_per_elem[old_elem_id])
        
        # Remove duplicates and sort
        new_smp.nodes = sorted(set(new_smp.nodes))
        
        # Map elements
        for old_elem_id in smp.elements:
            if old_elem_id in old_to_new_elements:
                new_smp.elements.extend(old_to_new_elements[old_elem_id])
        new_smp.elements = sorted(new_smp.elements)
        
        # Map conditions
        for old_cond_id in smp.conditions:
            if old_cond_id in old_to_new_conditions:
                new_smp.conditions.extend(old_to_new_conditions[old_cond_id])
        new_smp.conditions = sorted(new_smp.conditions)
        
        refined.sub_model_parts[smp_name] = new_smp
    
    return refined


def write_mdpa(data: MdpaData, filename: str):
    """Write MDPA data to file."""
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

    print(f"Written: {filename}")


def refine_mdpa(input_file: str, output_file: str, subdivisions: int = 2):
    """Main function to refine an MDPA mesh file."""
    print(f"Reading: {input_file}")
    data = parse_mdpa(input_file)

    print(f"Original mesh: {len(data.nodes)} nodes, {len(data.elements)} elements, {len(data.conditions)} conditions")

    print(f"Refining with {subdivisions} subdivisions...")
    refined = refine_mesh(data, subdivisions)

    print(f"Refined mesh: {len(refined.nodes)} nodes, {len(refined.elements)} elements, {len(refined.conditions)} conditions")

    write_mdpa(refined, output_file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python mdpa_refiner.py <input.mdpa> <output.mdpa> [subdivisions]")
        print("  subdivisions: number of subdivisions per element (default: 2)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    subdivisions = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    refine_mdpa(input_file, output_file, subdivisions)