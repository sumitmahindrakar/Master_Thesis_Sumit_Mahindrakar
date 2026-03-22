from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set


@dataclass
class SubModelPart:
    name: str
    nodes: List[int] = field(default_factory=list)
    elements: List[int] = field(default_factory=list)
    conditions: List[int] = field(default_factory=list)
    sub_model_parts: Dict[str, 'SubModelPart'] = field(default_factory=dict)


@dataclass
class MdpaData:
    header_lines: List[str] = field(default_factory=list)
    model_part_data: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    elements: Dict[int, dict] = field(default_factory=dict)
    element_type: str = ""
    elemental_data: Dict[str, Dict[int, str]] = field(default_factory=dict)
    conditions: Dict[int, dict] = field(default_factory=dict)
    condition_type: str = ""
    sub_model_parts: Dict[str, SubModelPart] = field(default_factory=dict)


def _parse_sub_model_part(lines, i):
    line = lines[i].strip()
    smp_name = line.split("Begin SubModelPart")[1].strip().split("//")[0].strip()
    smp = SubModelPart(name=smp_name)
    i += 1
    while i < len(lines):
        smp_line = lines[i].strip()
        if smp_line.startswith("End SubModelPart"):
            i += 1
            return smp, i
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
        if smp_line.startswith("Begin SubModelPart"):
            child_smp, i = _parse_sub_model_part(lines, i)
            smp.sub_model_parts[child_smp.name] = child_smp
            continue
        i += 1
    return smp, i


def parse_mdpa(filename: str) -> MdpaData:
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = MdpaData()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("Begin ModelPartData"):
            while i < len(lines) and not lines[i].strip().startswith("End ModelPartData"):
                data.model_part_data.append(lines[i])
                i += 1
            data.model_part_data.append(lines[i])
            i += 1
            continue
        if line.startswith("Begin Properties"):
            while i < len(lines) and not lines[i].strip().startswith("End Properties"):
                data.properties.append(lines[i])
                i += 1
            data.properties.append(lines[i])
            i += 1
            continue
        if line.startswith("Begin Nodes"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End Nodes"):
                node_line = lines[i].strip()
                if node_line and not node_line.startswith("//"):
                    parts = node_line.split()
                    if len(parts) >= 4:
                        data.nodes[int(parts[0])] = (
                            float(parts[1]), float(parts[2]), float(parts[3])
                        )
                i += 1
            i += 1
            continue
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
                        data.elements[int(parts[0])] = {
                            'property': int(parts[1]),
                            'nodes': [int(p) for p in parts[2:]]
                        }
                i += 1
            i += 1
            continue
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
                        data.elemental_data[data_name][int(parts[0])] = parts[1]
                i += 1
            i += 1
            continue
        if line.startswith("Begin Conditions"):
            parts = line.split("Begin Conditions")
            if len(parts) > 1:
                data.condition_type = parts[1].strip().split("//")[0].strip()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("End Conditions"):
                cond_line = lines[i].strip()
                if cond_line and not cond_line.startswith("//"):
                    parts = cond_line.split()
                    if len(parts) >= 3:
                        data.conditions[int(parts[0])] = {
                            'property': int(parts[1]),
                            'nodes': [int(p) for p in parts[2:]]
                        }
                i += 1
            i += 1
            continue
        if line.startswith("Begin SubModelPart"):
            smp, i = _parse_sub_model_part(lines, i)
            data.sub_model_parts[smp.name] = smp
            continue
        i += 1
    return data


def _find_support_nodes(data: MdpaData) -> Set[int]:
    """Auto-detect support nodes from DISPLACEMENT_support."""
    support: Set[int] = set()
    for name, smp in data.sub_model_parts.items():
        if "DISPLACEMENT" in name:
            support.update(smp.nodes)
    return support


def refine_mesh(
    data: MdpaData,
    subdivisions: int,
    beam_element_ids: Optional[List[int]] = None
) -> MdpaData:
    """
    Refine mesh. Point loads only on beam element nodes, supports excluded.

    Args:
        data: parsed MDPA
        subdivisions: per element
        beam_element_ids: template element IDs that are beams.
            If provided → conditions only on these elements' nodes
            If None → conditions on ALL nodes (except supports)
    """
    original_elem_ids = sorted(data.elements.keys())

    refined = MdpaData()
    refined.header_lines = data.header_lines.copy()
    refined.model_part_data = data.model_part_data.copy()
    refined.properties = data.properties.copy()
    refined.element_type = data.element_type
    refined.condition_type = "PointLoadCondition3D1N"

    support_nodes = _find_support_nodes(data)
    if support_nodes:
        print(f"      Support nodes (auto): {sorted(support_nodes)}")

    if beam_element_ids:
        print(f"      Beam elements (config): {beam_element_ids}")
    else:
        print(f"      Beam elements: ALL (no filter)")

    smps_with_conditions = {
        name for name, smp in data.sub_model_parts.items() if smp.conditions
    }

    # ── 1. Nodes ──
    node_map: Dict[int, int] = {}
    intermediates: Dict[int, List[int]] = {}
    next_id = 1
    seen: Set[int] = set()

    for eid in original_elem_ids:
        n1, n2 = data.elements[eid]['nodes']
        if n1 not in seen:
            seen.add(n1)
            node_map[n1] = next_id
            refined.nodes[next_id] = data.nodes[n1]
            next_id += 1
        p1, p2 = data.nodes[n1], data.nodes[n2]
        mid_ids = []
        for j in range(1, subdivisions):
            t = j / subdivisions
            refined.nodes[next_id] = (
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1]),
                p1[2] + t * (p2[2] - p1[2])
            )
            mid_ids.append(next_id)
            next_id += 1
        intermediates[eid] = mid_ids
        if n2 not in seen:
            seen.add(n2)
            node_map[n2] = next_id
            refined.nodes[next_id] = data.nodes[n2]
            next_id += 1

    new_support = {node_map[n] for n in support_nodes if n in node_map}

    # ── 2. Elements ──
    next_eid = 1
    old_to_new: Dict[int, List[int]] = {}

    for eid in original_elem_ids:
        n1, n2 = data.elements[eid]['nodes']
        all_n = [node_map[n1]] + intermediates[eid] + [node_map[n2]]
        new_ids = []
        for j in range(subdivisions):
            refined.elements[next_eid] = {
                'property': data.elements[eid]['property'],
                'nodes': [all_n[j], all_n[j + 1]]
            }
            new_ids.append(next_eid)
            next_eid += 1
        old_to_new[eid] = new_ids

    for dname, edata in data.elemental_data.items():
        refined.elemental_data[dname] = {}
        for old_eid, val in edata.items():
            for new_eid in old_to_new.get(old_eid, []):
                refined.elemental_data[dname][new_eid] = val

    # ── 3. Conditions: beam nodes only, minus supports ──
    if beam_element_ids:
        beam_nodes: Set[int] = set()
        for old_eid in beam_element_ids:
            if old_eid in old_to_new:
                for new_eid in old_to_new[old_eid]:
                    beam_nodes.update(refined.elements[new_eid]['nodes'])
        condition_nodes = sorted(beam_nodes - new_support)
        print(f"      Beam nodes: {len(beam_nodes)}, "
              f"after excluding supports: {len(condition_nodes)}")
    else:
        condition_nodes = sorted(set(refined.nodes.keys()) - new_support)
        print(f"      All nodes minus supports: {len(condition_nodes)}")

    for i, nid in enumerate(condition_nodes, start=1):
        refined.conditions[i] = {'property': 0, 'nodes': [nid]}

    all_cond_ids = list(range(1, len(condition_nodes) + 1))

    # ── 4. SubModelParts ──
    for smp_name, smp in data.sub_model_parts.items():
        new_smp = SubModelPart(name=smp_name)
        for old_n in smp.nodes:
            if old_n in node_map:
                new_smp.nodes.append(node_map[old_n])
        for old_eid in smp.elements:
            if old_eid in old_to_new:
                new_smp.elements.extend(old_to_new[old_eid])
                for new_eid in old_to_new[old_eid]:
                    new_smp.nodes.extend(refined.elements[new_eid]['nodes'])
        if smp_name in smps_with_conditions:
            new_smp.conditions = list(all_cond_ids)
            new_smp.nodes.extend(condition_nodes)
        new_smp.nodes = sorted(set(new_smp.nodes))
        new_smp.elements = sorted(new_smp.elements)
        new_smp.conditions = sorted(new_smp.conditions)
        refined.sub_model_parts[smp_name] = new_smp

    return refined


def _write_smp(f, smp, indent=""):
    f.write(f"{indent}Begin SubModelPart {smp.name}\n")
    inner = indent + "    "
    inner2 = inner + "    "
    f.write(f"{inner}Begin SubModelPartNodes\n")
    for nid in sorted(smp.nodes):
        f.write(f"{inner2}{nid}\n")
    f.write(f"{inner}End SubModelPartNodes\n")
    f.write(f"{inner}Begin SubModelPartElements\n")
    for eid in sorted(smp.elements):
        f.write(f"{inner2}{eid}\n")
    f.write(f"{inner}End SubModelPartElements\n")
    f.write(f"{inner}Begin SubModelPartConditions\n")
    for cid in sorted(smp.conditions):
        f.write(f"{inner2}{cid}\n")
    f.write(f"{inner}End SubModelPartConditions\n")
    for child_name, child_smp in smp.sub_model_parts.items():
        f.write("\n")
        _write_smp(f, child_smp, inner)
    f.write(f"{indent}End SubModelPart\n")


def write_mdpa(data: MdpaData, filename: str):
    with open(filename, 'w') as f:
        if data.model_part_data:
            for line in data.model_part_data:
                f.write(line if line.endswith('\n') else line + '\n')
        else:
            f.write("Begin ModelPartData\nEnd ModelPartData\n")
        f.write("\n")
        if data.properties:
            for line in data.properties:
                f.write(line if line.endswith('\n') else line + '\n')
        else:
            f.write("Begin Properties 0\nEnd Properties\n")
        f.write("\n")
        f.write("Begin Nodes\n")
        for nid in sorted(data.nodes.keys()):
            x, y, z = data.nodes[nid]
            f.write(f"    {nid}   {x:.10f}   {y:.10f}   {z:.10f}\n")
        f.write("End Nodes\n\n")
        f.write(f"Begin Elements {data.element_type}\n")
        for eid in sorted(data.elements.keys()):
            e = data.elements[eid]
            f.write(f"    {eid}   {e['property']}   "
                    f"{'   '.join(str(n) for n in e['nodes'])}\n")
        f.write("End Elements\n\n")
        for dname, edata in data.elemental_data.items():
            f.write(f"Begin ElementalData {dname}\n")
            for eid in sorted(edata.keys()):
                f.write(f"    {eid} {edata[eid]}\n")
            f.write("End ElementalData\n\n")
        if data.conditions:
            f.write(f"Begin Conditions {data.condition_type}\n")
            for cid in sorted(data.conditions.keys()):
                c = data.conditions[cid]
                f.write(f"    {cid} {c['property']} "
                        f"{' '.join(str(n) for n in c['nodes'])}\n")
            f.write("End Conditions\n\n")
        for smp_name, smp in data.sub_model_parts.items():
            _write_smp(f, smp)
            f.write("\n")
    print(f"Written: {filename}")


def refine_mdpa(input_file, output_file, subdivisions=2, beam_element_ids=None):
    data = parse_mdpa(input_file)
    refined = refine_mesh(data, subdivisions, beam_element_ids)
    write_mdpa(refined, output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python mdpa_refiner.py <input> <output> [subdivisions]")
        sys.exit(1)
    refine_mdpa(sys.argv[1], sys.argv[2],
                int(sys.argv[3]) if len(sys.argv) > 3 else 2)