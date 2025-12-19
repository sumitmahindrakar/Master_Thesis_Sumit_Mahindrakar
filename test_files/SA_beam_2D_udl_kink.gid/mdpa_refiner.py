"""
MDPA Mesh Refiner - Handles both 1-node and 2-node conditions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
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
    return 2  # Default to 2 nodes


def parse_mdpa(filename: str) -> MdpaData:
    """Parse an MDPA file and extract all data."""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = MdpaData()
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # ModelPartData
        if line.startswith("Begin ModelPartData"):
            while i < len(lines) and not lines[i].strip().startswith("End ModelPartData"):
                data.model_part_data.append(lines[i])
                i += 1
            data.model_part_data.append(lines[i])  # Include End line
            i += 1
            continue
        
        # Properties
        if line.startswith("Begin Properties"):
            while i < len(lines) and not lines[i].strip().startswith("End Properties"):
                data.properties.append(lines[i])
                i += 1
            data.properties.append(lines[i])  # Include End line
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
            # Extract element type
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
            # Extract condition type
            parts = line.split("Begin Conditions")
            if len(parts) > 1:
                data.condition_type = parts[1].strip().split("//")[0].strip()
                data.condition_num_nodes = detect_condition_nodes(data.condition_type)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End Conditions"):
                cond_line = lines[i].strip()
                if cond_line and not cond_line.startswith("//"):
                    parts = cond_line.split()
                    # Condition format: id property_id node1 [node2] [node3]...
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
                
                # SubModelPartNodes
                if smp_line.startswith("Begin SubModelPartNodes"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("End SubModelPartNodes"):
                        node_line = lines[i].strip()
                        if node_line and not node_line.startswith("//"):
                            smp.nodes.append(int(node_line))
                        i += 1
                    i += 1
                    continue
                
                # SubModelPartElements
                if smp_line.startswith("Begin SubModelPartElements"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("End SubModelPartElements"):
                        elem_line = lines[i].strip()
                        if elem_line and not elem_line.startswith("//"):
                            smp.elements.append(int(elem_line))
                        i += 1
                    i += 1
                    continue
                
                # SubModelPartConditions
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
    """Refine mesh by subdividing elements."""
    
    refined = MdpaData()
    refined.header_lines = data.header_lines.copy()
    refined.model_part_data = data.model_part_data.copy()
    refined.properties = data.properties.copy()
    refined.element_type = data.element_type
    refined.condition_type = data.condition_type
    refined.condition_num_nodes = data.condition_num_nodes
    
    # Copy original nodes
    refined.nodes = dict(data.nodes)
    
    # Track new nodes created for each element
    element_new_nodes = {}  # elem_id -> list of new node ids
    next_node_id = max(data.nodes.keys()) + 1
    
    # Create intermediate nodes for each element
    for elem_id, elem in data.elements.items():
        n1, n2 = elem['nodes']
        p1 = data.nodes[n1]
        p2 = data.nodes[n2]
        
        new_nodes = []
        for j in range(1, subdivisions):
            t = j / subdivisions
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            z = p1[2] + t * (p2[2] - p1[2])
            refined.nodes[next_node_id] = (x, y, z)
            new_nodes.append(next_node_id)
            next_node_id += 1
        
        element_new_nodes[elem_id] = new_nodes
    
    # Create refined elements
    next_elem_id = 1
    old_to_new_elements = {}  # old_elem_id -> list of new_elem_ids
    
    for elem_id, elem in data.elements.items():
        n1, n2 = elem['nodes']
        new_nodes = element_new_nodes[elem_id]
        all_nodes = [n1] + new_nodes + [n2]
        
        new_elem_ids = []
        for j in range(subdivisions):
            refined.elements[next_elem_id] = {
                'property': elem['property'],
                'nodes': [all_nodes[j], all_nodes[j + 1]]
            }
            new_elem_ids.append(next_elem_id)
            next_elem_id += 1
        
        old_to_new_elements[elem_id] = new_elem_ids
    
    # Handle elemental data
    for data_name, elem_data in data.elemental_data.items():
        refined.elemental_data[data_name] = {}
        for old_elem_id, value in elem_data.items():
            for new_elem_id in old_to_new_elements.get(old_elem_id, []):
                refined.elemental_data[data_name][new_elem_id] = value
    
    # Handle conditions based on type
    next_cond_id = 1
    old_to_new_conditions = {}  # old_cond_id -> list of new_cond_ids
    
    for cond_id, cond in data.conditions.items():
        if data.condition_num_nodes == 1:
            # Point condition - don't subdivide, just copy
            refined.conditions[next_cond_id] = {
                'property': cond['property'],
                'nodes': cond['nodes'].copy()
            }
            old_to_new_conditions[cond_id] = [next_cond_id]
            next_cond_id += 1
        else:
            # Line condition - find which element it belongs to and subdivide
            n1, n2 = cond['nodes']
            
            # Find the element with these nodes
            found_elem = None
            for elem_id, elem in data.elements.items():
                if set(elem['nodes']) == {n1, n2}:
                    found_elem = elem_id
                    break
            
            if found_elem is not None:
                new_nodes = element_new_nodes[found_elem]
                # Ensure correct ordering
                if data.elements[found_elem]['nodes'][0] == n1:
                    all_nodes = [n1] + new_nodes + [n2]
                else:
                    all_nodes = [n2] + new_nodes + [n1]
                
                new_cond_ids = []
                for j in range(subdivisions):
                    refined.conditions[next_cond_id] = {
                        'property': cond['property'],
                        'nodes': [all_nodes[j], all_nodes[j + 1]]
                    }
                    new_cond_ids.append(next_cond_id)
                    next_cond_id += 1
                
                old_to_new_conditions[cond_id] = new_cond_ids
            else:
                # Element not found, just copy the condition
                refined.conditions[next_cond_id] = {
                    'property': cond['property'],
                    'nodes': cond['nodes'].copy()
                }
                old_to_new_conditions[cond_id] = [next_cond_id]
                next_cond_id += 1
    
    # Handle SubModelParts
    for smp_name, smp in data.sub_model_parts.items():
        new_smp = SubModelPart(name=smp_name)
        
        # Add original nodes
        new_smp.nodes = list(smp.nodes)
        
        # Add new intermediate nodes from elements in this SMP
        for old_elem_id in smp.elements:
            if old_elem_id in element_new_nodes:
                new_smp.nodes.extend(element_new_nodes[old_elem_id])
        
        # Remove duplicates and sort
        new_smp.nodes = sorted(set(new_smp.nodes))
        
        # Map elements
        for old_elem_id in smp.elements:
            if old_elem_id in old_to_new_elements:
                new_smp.elements.extend(old_to_new_elements[old_elem_id])
        
        # Map conditions
        for old_cond_id in smp.conditions:
            if old_cond_id in old_to_new_conditions:
                new_smp.conditions.extend(old_to_new_conditions[old_cond_id])
        
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
        
        # Conditions (only if there are conditions)
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
            
            # Nodes
            if smp.nodes:
                f.write("    Begin SubModelPartNodes\n")
                for node_id in sorted(smp.nodes):
                    f.write(f"        {node_id}\n")
                f.write("    End SubModelPartNodes\n")
            
            # Elements
            if smp.elements:
                f.write("    Begin SubModelPartElements\n")
                for elem_id in sorted(smp.elements):
                    f.write(f"        {elem_id}\n")
                f.write("    End SubModelPartElements\n")
            
            # Conditions - THIS IS THE KEY FIX
            if smp.conditions:
                f.write("    Begin SubModelPartConditions\n")
                for cond_id in sorted(smp.conditions):
                    f.write(f"        {cond_id}\n")
                f.write("    End SubModelPartConditions\n")
            
            f.write("End SubModelPart\n\n")
    
    print(f"Written: {filename}")