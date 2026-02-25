"""
=================================================================
STEP 3c: EQUIVALENT NODAL LOADS FROM DISTRIBUTED LINE_LOAD
=================================================================

YOUR LINE_LOAD is applied in GLOBAL coordinates per node:
  Node 0 (base):   [0, 0, 0]      ← no load at support
  Node 1:          [0, 0, -37.86]  ← UDL in global Z direction
  Node 2:          [0, 0, -37.86]
  ...

For the PHYSICS LOSS, we need the equivalent nodal forces
that the distributed load produces at each element's ends.

For uniformly distributed load w on a beam of length L:

LOCAL COORDS (along beam):
  ┌──────────────────────────────────┐
  │         w (transverse)           │
  │  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  │
  │  ●══════════════════════════●    │
  │  n1          L            n2     │
  │                                  │
  │  Equivalent forces at ends:      │
  │    V1 = wL/2    V2 = wL/2       │ transverse force
  │    M1 = +wL²/12  M2 = -wL²/12   │ moments
  │                                  │
  │  If w has axial component too:   │
  │    N1 = w_axial*L/2              │ axial force
  │    N2 = w_axial*L/2              │
  └──────────────────────────────────┘

STEPS:
  1. For each element, average LINE_LOAD at its 2 nodes
     → element-level distributed load in GLOBAL coords
  
  2. Transform global load to LOCAL coords using T
     → get axial and transverse components
  
  3. Compute equivalent nodal forces in LOCAL coords
     → wL/2, ±wL²/12
  
  4. Transform back to GLOBAL coords using T^T
     → equivalent nodal forces in global XZ coords
  
  5. Assemble into per-node external force vector
     → each node sums contributions from its elements
=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import numpy as np
import os
from typing import Tuple


class EquivalentNodalLoads:
    """
    Computes equivalent nodal loads from LINE_LOAD for XZ plane frame.
    
    All operations differentiable (though loads are typically fixed
    inputs, keeping differentiability maintains clean computation graph).
    """

    @staticmethod
    def element_distributed_load(line_load: torch.Tensor,
                                  connectivity: torch.Tensor
                                  ) -> torch.Tensor:
        """
        Compute distributed load per element by averaging
        LINE_LOAD at its two end nodes.
        
        Args:
            line_load:    (N, 3) LINE_LOAD per node [wx, wy, wz]
            connectivity: (E, 2) element-to-node mapping
        
        Returns:
            w_elem: (E, 3) average distributed load per element
                    in GLOBAL coordinates
        
        Example for your frame:
            Element 0 (node 0→1):
              w_node0 = [0, 0, 0]
              w_node1 = [0, 0, -37.86]
              w_elem0 = [0, 0, -18.93]  ← average
            
            Element 1 (node 1→2):
              w_node1 = [0, 0, -37.86]
              w_node2 = [0, 0, -37.86]
              w_elem1 = [0, 0, -37.86]  ← same (both loaded)
        """
        n1 = connectivity[:, 0]
        n2 = connectivity[:, 1]
        w_elem = (line_load[n1] + line_load[n2]) / 2.0
        return w_elem

    @staticmethod
    def global_to_local_load(w_global: torch.Tensor,
                              cos_theta: torch.Tensor,
                              sin_theta: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform distributed load from global XZ to local coords.
        
        LOCAL coordinates:
          axial      = along element direction
          transverse = perpendicular to element (in XZ plane)
        
        For element at angle θ from X axis:
          w_axial      =  wx*cos(θ) + wz*sin(θ)
          w_transverse = -wx*sin(θ) + wz*cos(θ)
        
        Args:
            w_global:  (E, 3) distributed load in global [wx, wy, wz]
            cos_theta: (E,)   cos of element angle
            sin_theta: (E,)   sin of element angle
        
        Returns:
            w_axial:      (E,) axial component
            w_transverse: (E,) transverse component
        
        Example for your frame:
            Column (c=0, s=1), load [0,0,-37.86]:
              w_axial      = 0*0 + (-37.86)*1 = -37.86
                             (compression along column)
              w_transverse = -0*1 + (-37.86)*0 = 0
                             (no transverse load on columns)
            
            Beam (c=1, s=0), load [0,0,-37.86]:
              w_axial      = 0*1 + (-37.86)*0 = 0
                             (no axial load on beams)
              w_transverse = -0*0 + (-37.86)*1 = -37.86
                             (full transverse load on beams, downward)
              ✓ Makes sense physically!
        """
        wx = w_global[:, 0]  # X component
        wz = w_global[:, 2]  # Z component (Y=0 for XZ plane)

        w_axial = wx * cos_theta + wz * sin_theta
        w_transverse = -wx * sin_theta + wz * cos_theta

        return w_axial, w_transverse

    @staticmethod
    def equivalent_forces_local(w_axial: torch.Tensor,
                                 w_transverse: torch.Tensor,
                                 L: torch.Tensor
                                 ) -> torch.Tensor:
        """
        Compute equivalent nodal forces in LOCAL coordinates.
        
        For uniformly distributed load on beam element:
        
          Axial force:      N1 = w_axial * L / 2
                            N2 = w_axial * L / 2
          
          Transverse force: V1 = w_transverse * L / 2
                            V2 = w_transverse * L / 2
          
          Moments:          M1 = +w_transverse * L² / 12
                            M2 = -w_transverse * L² / 12
        
        Args:
            w_axial:      (E,) axial distributed load
            w_transverse: (E,) transverse distributed load
            L:            (E,) element lengths
        
        Returns:
            f_equiv_local: (E, 6) equivalent nodal forces in local coords
                           [N1, V1, M1, N2, V2, M2]
        """
        E = L.shape[0]
        f = torch.zeros(E, 6, device=L.device, dtype=L.dtype)

        # Node 1
        f[:, 0] = w_axial * L / 2.0          # N1
        f[:, 1] = w_transverse * L / 2.0      # V1
        f[:, 2] = w_transverse * L.pow(2) / 12.0  # M1

        # Node 2
        f[:, 3] = w_axial * L / 2.0           # N2
        f[:, 4] = w_transverse * L / 2.0       # V2
        f[:, 5] = -w_transverse * L.pow(2) / 12.0  # M2

        return f

    @staticmethod
    def local_to_global_forces(f_local: torch.Tensor,
                                cos_theta: torch.Tensor,
                                sin_theta: torch.Tensor
                                ) -> torch.Tensor:
        """
        Transform equivalent nodal forces from local to global coords.
        
        f_global = T^T × f_local
        
        For XZ plane:
          Fx_global =  c * N_local + (-s) * V_local
          Fz_global =  s * N_local +   c  * V_local
          My_global =  M_local  (unchanged)
        
        Args:
            f_local:   (E, 6) local forces [N1,V1,M1,N2,V2,M2]
            cos_theta: (E,) cos of element angle
            sin_theta: (E,) sin of element angle
        
        Returns:
            f_global: (E, 6) global forces
                      [Fx1, Fz1, My1, Fx2, Fz2, My2]
        """
        c = cos_theta
        s = sin_theta

        f_global = torch.zeros_like(f_local)

        # Node 1: T^T × [N1, V1, M1]
        f_global[:, 0] =  c * f_local[:, 0] - s * f_local[:, 1]  # Fx1
        f_global[:, 1] =  s * f_local[:, 0] + c * f_local[:, 1]  # Fz1
        f_global[:, 2] =  f_local[:, 2]                           # My1

        # Node 2: T^T × [N2, V2, M2]
        f_global[:, 3] =  c * f_local[:, 3] - s * f_local[:, 4]  # Fx2
        f_global[:, 4] =  s * f_local[:, 3] + c * f_local[:, 4]  # Fz2
        f_global[:, 5] =  f_local[:, 5]                           # My2

        return f_global

    def assemble_nodal_forces(self,
                               f_global_elem: torch.Tensor,
                               connectivity: torch.Tensor,
                               n_nodes: int
                               ) -> torch.Tensor:
        """
        Assemble element equivalent forces into per-node force vector.
        
        Each node receives contributions from ALL connected elements.
        
        Args:
            f_global_elem: (E, 6) global equiv forces per element
                           [Fx1, Fz1, My1, Fx2, Fz2, My2]
            connectivity:  (E, 2) element-to-node mapping
            n_nodes:       total number of nodes
        
        Returns:
            F_ext: (N, 3) external force at each node
                   [Fx, Fz, My] in global coordinates
        
        Example for node 3 (connected to elements 2, 3, 14):
            F_ext[3] = f_elem2_at_n2 + f_elem3_at_n1 + f_elem14_at_n1
        """
        F_ext = torch.zeros(n_nodes, 3,
                             device=f_global_elem.device,
                             dtype=f_global_elem.dtype)

        n1 = connectivity[:, 0]  # (E,) start nodes
        n2 = connectivity[:, 1]  # (E,) end nodes

        # Scatter-add node 1 contributions
        # f_global_elem[:, 0:3] = [Fx1, Fz1, My1]
        F_ext.scatter_add_(0,
                            n1.unsqueeze(1).expand(-1, 3),
                            f_global_elem[:, 0:3])

        # Scatter-add node 2 contributions
        # f_global_elem[:, 3:6] = [Fx2, Fz2, My2]
        F_ext.scatter_add_(0,
                            n2.unsqueeze(1).expand(-1, 3),
                            f_global_elem[:, 3:6])

        return F_ext

    # ─────────────────────────────────────────────
    # MAIN COMPUTATION: from PyG Data
    # ─────────────────────────────────────────────

    def compute_from_data(self, data) -> dict:
        """
        Compute all equivalent nodal loads from PyG Data object.
        
        Args:
            data: PyG Data with:
                .line_load       (N, 3) distributed load per node
                .connectivity    (E, 2) element connectivity
                .elem_lengths    (E,)   element lengths
                .elem_directions (E, 3) element direction vectors
        
        Returns:
            dict with:
                'F_ext':           (N, 3) total external force per node
                                   [Fx, Fz, My] in global coords
                'f_equiv_local':   (E, 6) equiv forces in local coords
                'f_equiv_global':  (E, 6) equiv forces in global coords
                'w_axial':         (E,)   axial load component
                'w_transverse':    (E,)   transverse load component
                'w_elem':          (E, 3) average element load
        """
        dirs = data.elem_directions
        cos_theta = dirs[:, 0]
        sin_theta = dirs[:, 2]
        L = data.elem_lengths
        N = data.line_load.shape[0]

        # Step 1: Element distributed load (average of end nodes)
        w_elem = self.element_distributed_load(
            data.line_load, data.connectivity)

        # Step 2: Transform to local coords
        w_axial, w_transverse = self.global_to_local_load(
            w_elem, cos_theta, sin_theta)

        # Step 3: Equivalent forces in local coords
        f_equiv_local = self.equivalent_forces_local(
            w_axial, w_transverse, L)

        # Step 4: Transform to global coords
        f_equiv_global = self.local_to_global_forces(
            f_equiv_local, cos_theta, sin_theta)

        # Step 5: Assemble into per-node force vector
        F_ext = self.assemble_nodal_forces(
            f_equiv_global, data.connectivity, N)

        return {
            'F_ext':          F_ext,           # (N, 3) MAIN OUTPUT
            'f_equiv_local':  f_equiv_local,   # (E, 6)
            'f_equiv_global': f_equiv_global,  # (E, 6)
            'w_axial':        w_axial,         # (E,)
            'w_transverse':   w_transverse,    # (E,)
            'w_elem':         w_elem,          # (E, 3)
        }


# ================================================================
# VERIFICATION
# ================================================================

def verify_equivalent_loads():
    """
    Verify equivalent nodal load computation.
    """
    print(f"\n{'═'*60}")
    print(f"  EQUIVALENT NODAL LOADS VERIFICATION")
    print(f"{'═'*60}")

    enl = EquivalentNodalLoads()

    # ─── Test 1: Horizontal beam with vertical UDL ───
    print(f"\n  Test 1: Horizontal beam, vertical UDL")
    print(f"  w = -40 N/m in Z, L = 6 m, beam along X")

    w = torch.tensor([[-40.0]])  # w_transverse
    L = torch.tensor([6.0])
    
    # For horizontal beam: cos=1, sin=0
    # w_axial = 0, w_transverse = -40
    w_global = torch.tensor([[0.0, 0.0, -40.0]])
    cos_t = torch.tensor([1.0])
    sin_t = torch.tensor([0.0])

    w_ax, w_tr = enl.global_to_local_load(w_global, cos_t, sin_t)
    print(f"    w_axial = {w_ax.item():.2f} (expected 0)")
    print(f"    w_transverse = {w_tr.item():.2f} (expected -40)")

    f_local = enl.equivalent_forces_local(w_ax, w_tr, L)
    print(f"\n    Equivalent local forces:")
    print(f"    N1={f_local[0,0]:.2f}, V1={f_local[0,1]:.2f}, "
          f"M1={f_local[0,2]:.2f}")
    print(f"    N2={f_local[0,3]:.2f}, V2={f_local[0,4]:.2f}, "
          f"M2={f_local[0,5]:.2f}")
    print(f"    Expected: V1=V2={-40*6/2:.2f}, "
          f"M1={-40*36/12:.2f}, M2={40*36/12:.2f}")

    f_global = enl.local_to_global_forces(f_local, cos_t, sin_t)
    print(f"\n    Equivalent global forces (should be same for horizontal):")
    print(f"    Fx1={f_global[0,0]:.2f}, Fz1={f_global[0,1]:.2f}, "
          f"My1={f_global[0,2]:.2f}")
    print(f"    Fx2={f_global[0,3]:.2f}, Fz2={f_global[0,4]:.2f}, "
          f"My2={f_global[0,5]:.2f}")

    # ─── Test 2: Vertical column with vertical UDL ───
    print(f"\n  Test 2: Vertical column, vertical UDL")
    print(f"  w = -40 N/m in Z, L = 3 m, column along Z")

    w_global_col = torch.tensor([[0.0, 0.0, -40.0]])
    cos_col = torch.tensor([0.0])
    sin_col = torch.tensor([1.0])

    w_ax_c, w_tr_c = enl.global_to_local_load(
        w_global_col, cos_col, sin_col)
    print(f"    w_axial = {w_ax_c.item():.2f} (expected -40, "
          f"compression along column)")
    print(f"    w_transverse = {w_tr_c.item():.2f} (expected 0)")

    f_local_c = enl.equivalent_forces_local(w_ax_c, w_tr_c, 
                                             torch.tensor([3.0]))
    print(f"\n    Equivalent local forces:")
    print(f"    N1={f_local_c[0,0]:.2f} (axial, expected {-40*3/2:.2f})")
    print(f"    V1={f_local_c[0,1]:.2f} (transverse, expected 0)")
    print(f"    M1={f_local_c[0,2]:.2f} (moment, expected 0)")

    f_global_c = enl.local_to_global_forces(
        f_local_c, cos_col, sin_col)
    print(f"\n    Equivalent global forces:")
    print(f"    Fx1={f_global_c[0,0]:.2f} (expected 0)")
    print(f"    Fz1={f_global_c[0,1]:.2f} "
          f"(expected {-40*3/2:.2f}, downward)")

    # ─── Test 3: Global equilibrium ───
    print(f"\n  Test 3: Simply supported beam — total load check")
    print(f"  Two-node beam, UDL = -40 N/m, L = 6 m")
    print(f"  Total applied load = wL = {-40*6:.1f} N")

    # Assemble for two-node beam
    line_load = torch.tensor([[0.0, 0.0, -40.0],
                               [0.0, 0.0, -40.0]])
    conn = torch.tensor([[0, 1]])

    class FakeData:
        pass
    data = FakeData()
    data.line_load = line_load
    data.connectivity = conn
    data.elem_lengths = torch.tensor([6.0])
    data.elem_directions = torch.tensor([[1.0, 0.0, 0.0]])

    result = enl.compute_from_data(data)
    F_ext = result['F_ext']
    total_Fz = F_ext[:, 1].sum()
    print(f"    F_ext node 0: {F_ext[0].tolist()}")
    print(f"    F_ext node 1: {F_ext[1].tolist()}")
    print(f"    Total Fz = {total_Fz.item():.1f} "
          f"(expected {-40*6:.1f}) "
          f"{'✓' if abs(total_Fz.item() + 240) < 0.1 else '✗'}")

    total_moment = (F_ext[:, 2].sum() + 
                    F_ext[1, 1] * 6.0)  # moment about node 0
    print(f"    Total moment about node 0 = {total_moment.item():.1f} "
          f"(should be 0 for uniform load) "
          f"{'✓' if abs(total_moment.item()) < 0.1 else '✗'}")

    print(f"\n{'═'*60}\n")


def test_with_frame_data():
    """
    Test with your actual frame data.
    """
    print(f"\n{'═'*60}")
    print(f"  TEST WITH YOUR FRAME DATA")
    print(f"{'═'*60}")

    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]

    enl = EquivalentNodalLoads()
    result = enl.compute_from_data(data)

    F_ext = result['F_ext']
    w_elem = result['w_elem']
    w_ax = result['w_axial']
    w_tr = result['w_transverse']
    f_local = result['f_equiv_local']
    f_global = result['f_equiv_global']
    
    conn = data.connectivity
    dirs = data.elem_directions

    # Element classification
    print(f"\n  Element load decomposition:")
    print(f"  {'Elem':>4} {'Type':>8} {'w_global_z':>10} "
          f"{'w_axial':>10} {'w_transverse':>12}")
    for e in range(min(data.n_elements, 36)):
        dx = abs(dirs[e, 0].item())
        dz = abs(dirs[e, 2].item())
        etype = 'COLUMN' if dz > dx else 'BEAM'
        n1, n2 = conn[e]
        print(f"  {e:>4} {etype:>8} {w_elem[e,2]:>10.2f} "
              f"{w_ax[e]:>10.2f} {w_tr[e]:>12.2f}")
        if e == 5 or e == 11:
            print(f"  {'---':>4}")

    # Per-node external forces
    print(f"\n  External force vector F_ext (N, 3) = [Fx, Fz, My]:")
    print(f"  {'Node':>4} {'Fx':>12} {'Fz':>12} {'My':>12}")
    for n in range(data.num_nodes):
        f = F_ext[n]
        marker = ' ← support' if data.bc_disp[n, 0] > 0.5 else ''
        print(f"  {n:>4} {f[0]:>12.2f} {f[1]:>12.2f} "
              f"{f[2]:>12.2f}{marker}")

    # Global equilibrium
    total_Fx = F_ext[:, 0].sum()
    total_Fz = F_ext[:, 1].sum()
    total_My = F_ext[:, 2].sum()
    print(f"\n  Global force sums:")
    print(f"    ΣFx = {total_Fx.item():.4f}")
    print(f"    ΣFz = {total_Fz.item():.4f}")
    print(f"    ΣMy = {total_My.item():.4f}")
    
    # Expected total vertical load
    # Each column element has LINE_LOAD contributing axially
    # Each beam element has LINE_LOAD contributing transversely
    total_line = data.line_load[:, 2].sum()
    print(f"\n  Total LINE_LOAD (Z): {total_line.item():.2f}")
    print(f"  Total equivalent Fz: {total_Fz.item():.2f}")

    print(f"\n{'═'*60}\n")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 3c: Equivalent Nodal Loads")
    print("=" * 60)

    verify_equivalent_loads()

    if os.path.exists("DATA/graph_dataset.pt"):
        test_with_frame_data()

    print(f"{'='*60}")
    print(f"  STEP 3c COMPLETE ✓")
    print(f"")
    print(f"  EquivalentNodalLoads provides:")
    print(f"    compute_from_data(data) → dict with:")
    print(f"      F_ext:          (N, 3) [Fx, Fz, My] per node")
    print(f"      f_equiv_local:  (E, 6) local equiv forces")
    print(f"      f_equiv_global: (E, 6) global equiv forces")
    print(f"      w_axial:        (E,) axial load component")
    print(f"      w_transverse:   (E,) transverse load component")
    print(f"")
    print(f"  This F_ext is used in equilibrium loss:")
    print(f"    Σ f_internal_at_node = F_ext_at_node")
    print(f"")
    print(f"  Ready for Step 3d (Physics Loss Functions)")
    print(f"{'='*60}")