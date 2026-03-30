"""
=================================================================
postprocess.py — Extract Nodal Results from Trained PIGNN
=================================================================

Extracts per node:
  - Global displacements: ux, uz, θy  (with sign)
  - Global face forces:   Fx, Fz, My per face (with sign)
  - Local face forces:    Fx_loc, Fz_loc, My_loc per element end
  - Element end forces in local coordinates
  - Internal forces: N, V, M at element ends

USAGE:
  python postprocess.py
  
  Or from another script:
    from postprocess import PIGNNPostProcessor
    pp = PIGNNPostProcessor("RESULTS/best.pt", "DATA/graph_dataset.pt")
    results = pp.extract_all(graph_idx=0)
    pp.print_node_results(results, node_ids=[0, 1, 2])
=================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from model import PIGNN


class PIGNNPostProcessor:
    """
    Post-processor to extract forces, displacements, and 
    derived quantities from a trained PIGNN model.
    
    Provides results WITH CORRECT SIGNS following the convention:
      Element end forces (element → node):
        At A (s=0): Fx_A^loc = -N,  Fz_A^loc = -V,  My_A^loc = -M
        At B (s=L): Fx_B^loc = +N,  Fz_B^loc = +V,  My_B^loc = +M
    """

    def __init__(self, checkpoint_path, data_path,
                 hidden_dim=128, n_layers=6,
                 node_in_dim=9, edge_in_dim=10,
                 device='cpu'):
        """
        Args:
            checkpoint_path: path to saved model checkpoint (.pt)
            data_path:       path to graph dataset (.pt)
            hidden_dim:      must match training config
            n_layers:        must match training config
            node_in_dim:     must match training config
            edge_in_dim:     must match training config
            device:          'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # ── Load model ──
        self.model = PIGNN(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)
        
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        print(f"  ✓ Model loaded from {checkpoint_path}")
        print(f"    Trained for {ckpt['epoch']} epochs")
        print(f"    Final losses: {ckpt.get('losses', 'N/A')}")
        
        # ── Load data ──
        self.data_list = torch.load(data_path, weights_only=False)
        print(f"  ✓ Loaded {len(self.data_list)} graphs from {data_path}")

    # ════════════════════════════════════════════════════════
    # A. CORE EXTRACTION
    # ════════════════════════════════════════════════════════

    @torch.no_grad()
    def extract_all(self, graph_idx=0):
        """
        Run forward pass and extract ALL results for one graph.
        
        Args:
            graph_idx: index into the dataset
            
        Returns:
            dict with keys:
                'coords':            (N, 3) node coordinates [x, y, z]
                'n_nodes':           int
                'n_elements':        int
                
                # ── Global Displacements ──
                'disp_global':       (N, 3) [ux, uz, θy]
                
                # ── Global Face Forces ──
                'face_forces_global':(N, 4, 3) [Fx, Fz, My] per face
                'face_mask':         (N, 4) connectivity mask
                
                # ── Element End Forces (Global) ──
                'ff_A_global':       (E, 3) [Fx, Fz, My] at A-end (global)
                'ff_B_global':       (E, 3) [Fx, Fz, My] at B-end (global)
                
                # ── Element End Forces (Local) ──
                'ff_A_local':        (E, 3) [Fx_loc, Fz_loc, My_loc] at A
                'ff_B_local':        (E, 3) [Fx_loc, Fz_loc, My_loc] at B
                
                # ── Element End Displacements (Global) ──
                'disp_A_global':     (E, 3) [ux, uz, θy] at A-end
                'disp_B_global':     (E, 3) [ux, uz, θy] at B-end
                
                # ── Element End Displacements (Local) ──
                'disp_A_local':      (E, 3) [u_L, w_L, θ_L] at A
                'disp_B_local':      (E, 3) [u_L, w_L, θ_L] at B
                
                # ── Internal Forces (derived from local face forces) ──
                'N_A':               (E,) axial force at A  (N = -Fx_A^loc)
                'N_B':               (E,) axial force at B  (N = +Fx_B^loc)
                'V_A':               (E,) shear force at A  (V = -Fz_A^loc)
                'V_B':               (E,) shear force at B  (V = +Fz_B^loc)
                'M_A':               (E,) bending moment at A (M = -My_A^loc)
                'M_B':               (E,) bending moment at B (M = +My_B^loc)
                
                # ── Boundary Conditions ──
                'bc_disp':           (N, 1) displacement BC flag
                'bc_rot':            (N, 1) rotation BC flag
                'is_support':        (N,) bool
                
                # ── Element Properties ──
                'connectivity':      (E, 2) [nodeA, nodeB]
                'elem_directions':   (E, 3) unit direction vectors
                'elem_lengths':      (E,) element lengths
                'prop_E':            (E,) Young's modulus
                'prop_A':            (E,) cross-section area
                'prop_I':            (E,) moment of inertia
                
                # ── Ground Truth (if available) ──
                'true_disp':         (N, 3) Kratos displacement [ux, uz, θy]
                'disp_error':        (N, 3) pred - true
                
                # ── Reaction Forces (at supports) ──
                'reactions_global':  (N_sup, 3) [Rx, Rz, M_react]
                'support_node_ids':  list of support node indices
        """
        data = self.data_list[graph_idx].to(self.device)
        
        # ── Forward pass ──
        pred = self.model(data)  # (N, 15)
        
        # ── Extract predictions ──
        N = pred.shape[0]
        disp_global = pred[:, 0:3].cpu().numpy()            # (N, 3)
        face_forces_global = pred[:, 3:15].reshape(N, 4, 3).cpu().numpy()
        
        # ── Coordinates ──
        coords = data.coords.cpu().numpy()                   # (N, 3)
        
        # ── Connectivity and element data ──
        conn = data.connectivity.cpu().numpy()                # (E, 2)
        elem_dirs = data.elem_directions.cpu().numpy()        # (E, 3)
        E_count = conn.shape[0]
        
        # ── Compute element lengths ──
        coord_A = coords[conn[:, 0]]  # (E, 3)
        coord_B = coords[conn[:, 1]]  # (E, 3)
        elem_lengths = np.linalg.norm(coord_B - coord_A, axis=1)  # (E,)
        
        # ── Boundary conditions ──
        bc_disp = data.bc_disp.cpu().numpy()                  # (N, 1)
        bc_rot = data.bc_rot.cpu().numpy()                    # (N, 1)
        is_support = bc_disp.squeeze(-1) > 0.5                # (N,)
        
        # ── Face mask ──
        face_mask = data.face_mask.cpu().numpy()              # (N, 4)
        
        # ── Element properties ──
        prop_E_vals = data.prop_E.cpu().numpy()               # (E,)
        prop_A_vals = data.prop_A.cpu().numpy()               # (E,)
        prop_I_vals = data.prop_I22.cpu().numpy()             # (E,)
        
        # ════════════════════════════════════
        # ELEMENT END DATA (Global)
        # ════════════════════════════════════
        ff_A_global, ff_B_global = self._gather_element_end_forces(
            face_forces_global, data
        )  # each (E, 3)
        
        disp_A_global = disp_global[conn[:, 0]]  # (E, 3)
        disp_B_global = disp_global[conn[:, 1]]  # (E, 3)
        
        # ════════════════════════════════════
        # TRANSFORM TO LOCAL (Concept 2)
        # ════════════════════════════════════
        ff_A_local, ff_B_local = self._transform_to_local_np(
            ff_A_global, ff_B_global, elem_dirs
        )
        disp_A_local, disp_B_local = self._transform_to_local_np(
            disp_A_global, disp_B_global, elem_dirs
        )
        
        # ════════════════════════════════════
        # INTERNAL FORCES (from sign convention)
        # ════════════════════════════════════
        # At A (s=0): Fx_A^loc = -N  →  N = -Fx_A^loc
        #             Fz_A^loc = -V  →  V = -Fz_A^loc
        #             My_A^loc = -M  →  M = -My_A^loc
        # At B (s=L): Fx_B^loc = +N  →  N = +Fx_B^loc
        #             Fz_B^loc = +V  →  V = +Fz_B^loc
        #             My_B^loc = +M  →  M = +My_B^loc
        
        N_A = -ff_A_local[:, 0]   # (E,)
        N_B =  ff_B_local[:, 0]   # (E,)
        V_A = -ff_A_local[:, 1]   # (E,)
        V_B =  ff_B_local[:, 1]   # (E,)
        M_A = -ff_A_local[:, 2]   # (E,)
        M_B =  ff_B_local[:, 2]   # (E,)
        
        # ════════════════════════════════════
        # REACTION FORCES (at supports)
        # ════════════════════════════════════
        support_ids = np.where(is_support)[0]
        reactions_global = np.zeros((len(support_ids), 3))
        for i, nid in enumerate(support_ids):
            # Reaction = sum of all face forces at support node
            # (external equilibrium: reaction balances internal forces)
            reactions_global[i] = face_forces_global[nid].sum(axis=0)
        
        # ════════════════════════════════════
        # GROUND TRUTH COMPARISON
        # ════════════════════════════════════
        true_disp = None
        disp_error = None
        if hasattr(data, 'y_node') and data.y_node is not None:
            true_disp = data.y_node.cpu().numpy()            # (N, 3)
            disp_error = disp_global - true_disp             # (N, 3)
        
        # ════════════════════════════════════
        # ASSEMBLE RESULTS
        # ════════════════════════════════════
        results = {
            # Geometry
            'coords':             coords,
            'n_nodes':            N,
            'n_elements':         E_count,
            
            # Global displacements
            'disp_global':        disp_global,
            
            # Global face forces
            'face_forces_global': face_forces_global,
            'face_mask':          face_mask,
            
            # Element end forces (global)
            'ff_A_global':        ff_A_global,
            'ff_B_global':        ff_B_global,
            
            # Element end forces (local)
            'ff_A_local':         ff_A_local,
            'ff_B_local':         ff_B_local,
            
            # Element end displacements (global)
            'disp_A_global':      disp_A_global,
            'disp_B_global':      disp_B_global,
            
            # Element end displacements (local)
            'disp_A_local':       disp_A_local,
            'disp_B_local':       disp_B_local,
            
            # Internal forces
            'N_A': N_A, 'N_B': N_B,
            'V_A': V_A, 'V_B': V_B,
            'M_A': M_A, 'M_B': M_B,
            
            # Boundary conditions
            'bc_disp':            bc_disp,
            'bc_rot':             bc_rot,
            'is_support':         is_support,
            
            # Element properties
            'connectivity':       conn,
            'elem_directions':    elem_dirs,
            'elem_lengths':       elem_lengths,
            'prop_E':             prop_E_vals,
            'prop_A':             prop_A_vals,
            'prop_I':             prop_I_vals,
            
            # Ground truth
            'true_disp':          true_disp,
            'disp_error':         disp_error,
            
            # Reactions
            'reactions_global':   reactions_global,
            'support_node_ids':   support_ids.tolist(),
        }
        
        return results

    # ════════════════════════════════════════════════════════
    # B. HELPER METHODS
    # ════════════════════════════════════════════════════════

    def _gather_element_end_forces(self, face_forces_global, data):
        """
        Gather face forces at element ends A and B.
        
        Uses face_element_id and face_is_A_end to determine
        which face of which node corresponds to each element end.
        
        Args:
            face_forces_global: (N, 4, 3) numpy array
            data: PyG Data object
            
        Returns:
            ff_A: (E, 3) face forces at A-end [Fx, Fz, My] global
            ff_B: (E, 3) face forces at B-end [Fx, Fz, My] global
        """
        feid = data.face_element_id.cpu().numpy()   # (N, 4)
        faa  = data.face_is_A_end.cpu().numpy()      # (N, 4)
        fm   = data.face_mask.cpu().numpy()           # (N, 4)
        E = data.connectivity.shape[0]
        N = face_forces_global.shape[0]
        
        ff_A = np.zeros((E, 3))
        ff_B = np.zeros((E, 3))
        
        for f in range(4):
            for n in range(N):
                if fm[n, f] < 0.5:
                    continue
                e = int(feid[n, f])
                if e < 0:
                    continue
                if faa[n, f] == 1:
                    ff_A[e] = face_forces_global[n, f, :]
                else:
                    ff_B[e] = face_forces_global[n, f, :]
        
        return ff_A, ff_B

    def _transform_to_local_np(self, vec_A, vec_B, elem_directions):
        """
        Global → Local coordinate transformation (numpy).
        
        Rotation matrix T(α):
            [v_L0]   [ cos α   sin α   0] [v0]
            [v_L1] = [-sin α   cos α   0] [v1]
            [v_L2]   [ 0       0       1] [v2]
        
        For 2D XZ plane:
            cos α = elem_directions[:, 0]
            sin α = elem_directions[:, 2]
        
        Args:
            vec_A: (E, 3) global values at A [vx, vz, vθ]
            vec_B: (E, 3) global values at B [vx, vz, vθ]
            elem_directions: (E, 3)
            
        Returns:
            vec_A_loc: (E, 3) local values at A
            vec_B_loc: (E, 3) local values at B
        """
        cos_a = elem_directions[:, 0:1]  # (E, 1)
        sin_a = elem_directions[:, 2:3]  # (E, 1)
        
        def rotate(v):
            vx = v[:, 0:1]
            vz = v[:, 1:2]
            vt = v[:, 2:3]
            return np.concatenate([
                 vx * cos_a + vz * sin_a,
                -vx * sin_a + vz * cos_a,
                 vt,
            ], axis=1)
        
        return rotate(vec_A), rotate(vec_B)

    # ════════════════════════════════════════════════════════
    # C. PRINTING / DISPLAY METHODS
    # ════════════════════════════════════════════════════════

    def print_node_results(self, results, node_ids=None):
        """
        Print detailed results at specified nodes.
        
        Args:
            results: dict from extract_all()
            node_ids: list of node indices, or None for all nodes
        """
        if node_ids is None:
            node_ids = list(range(results['n_nodes']))
        
        print(f"\n{'═'*90}")
        print(f"  NODAL RESULTS — Global Displacements & Face Forces")
        print(f"{'═'*90}")
        
        face_names = ['+x', '-x', '+z', '-z']
        
        for nid in node_ids:
            coords = results['coords'][nid]
            is_sup = results['is_support'][nid]
            bc_d = results['bc_disp'][nid, 0]
            bc_r = results['bc_rot'][nid, 0]
            
            sup_type = ""
            if bc_d > 0.5 and bc_r > 0.5:
                sup_type = " [FIXED SUPPORT]"
            elif bc_d > 0.5:
                sup_type = " [PINNED SUPPORT]"
            elif bc_r > 0.5:
                sup_type = " [ROLLER (rot fixed)]"
            
            print(f"\n  ┌─── Node {nid}{sup_type}")
            print(f"  │  Coordinates: x={coords[0]:.4f}, "
                  f"y={coords[1]:.4f}, z={coords[2]:.4f}")
            
            # ── Global displacements ──
            ux = results['disp_global'][nid, 0]
            uz = results['disp_global'][nid, 1]
            th = results['disp_global'][nid, 2]
            print(f"  │")
            print(f"  │  Global Displacements:")
            print(f"  │    ux  = {ux:+14.6e}")
            print(f"  │    uz  = {uz:+14.6e}")
            print(f"  │    θy  = {th:+14.6e}")
            
            # ── Ground truth comparison ──
            if results['true_disp'] is not None:
                true_ux = results['true_disp'][nid, 0]
                true_uz = results['true_disp'][nid, 1]
                true_th = results['true_disp'][nid, 2]
                err_ux = results['disp_error'][nid, 0]
                err_uz = results['disp_error'][nid, 1]
                err_th = results['disp_error'][nid, 2]
                print(f"  │  Ground Truth Displacements:")
                print(f"  │    ux  = {true_ux:+14.6e}  "
                      f"(error: {err_ux:+.4e})")
                print(f"  │    uz  = {true_uz:+14.6e}  "
                      f"(error: {err_uz:+.4e})")
                print(f"  │    θy  = {true_th:+14.6e}  "
                      f"(error: {err_th:+.4e})")
            
            # ── Global face forces ──
            print(f"  │")
            print(f"  │  Global Face Forces:")
            print(f"  │  {'Face':<6} {'Mask':>4}  "
                  f"{'Fx':>14}  {'Fz':>14}  {'My':>14}")
            print(f"  │  {'─'*62}")
            for f in range(4):
                mask_val = results['face_mask'][nid, f]
                fx = results['face_forces_global'][nid, f, 0]
                fz = results['face_forces_global'][nid, f, 1]
                my = results['face_forces_global'][nid, f, 2]
                active = "●" if mask_val > 0.5 else "○"
                print(f"  │  {face_names[f]:<6} {active:>4}  "
                      f"{fx:+14.6e}  {fz:+14.6e}  {my:+14.6e}")
            
            # ── Equilibrium check ──
            sum_fx = results['face_forces_global'][nid, :, 0].sum()
            sum_fz = results['face_forces_global'][nid, :, 1].sum()
            sum_my = results['face_forces_global'][nid, :, 2].sum()
            print(f"  │  {'Sum':<6} {'':>4}  "
                  f"{sum_fx:+14.6e}  {sum_fz:+14.6e}  {sum_my:+14.6e}")
            
            print(f"  └{'─'*70}")

    def print_element_results(self, results, element_ids=None):
        """
        Print detailed element end results (local and global).
        
        Args:
            results: dict from extract_all()
            element_ids: list of element indices, or None for all
        """
        if element_ids is None:
            element_ids = list(range(results['n_elements']))
        
        print(f"\n{'═'*100}")
        print(f"  ELEMENT END RESULTS — Forces & Displacements")
        print(f"{'═'*100}")
        
        for eid in element_ids:
            nA = results['connectivity'][eid, 0]
            nB = results['connectivity'][eid, 1]
            L = results['elem_lengths'][eid]
            d = results['elem_directions'][eid]
            angle = np.degrees(np.arctan2(d[2], d[0]))
            
            E_val = results['prop_E'][eid]
            A_val = results['prop_A'][eid]
            I_val = results['prop_I'][eid]
            
            print(f"\n  ┌─── Element {eid}: "
                  f"Node {nA} → Node {nB}")
            print(f"  │  Length: {L:.4f}  "
                  f"Angle: {angle:.1f}°  "
                  f"Direction: [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]")
            print(f"  │  E={E_val:.4e}  A={A_val:.4e}  "
                  f"I={I_val:.4e}")
            
            # ── Global end forces ──
            print(f"  │")
            print(f"  │  Global End Forces:")
            print(f"  │  {'End':<5} {'Fx':>14}  {'Fz':>14}  {'My':>14}")
            print(f"  │  {'─'*52}")
            print(f"  │  {'A':<5} "
                  f"{results['ff_A_global'][eid, 0]:+14.6e}  "
                  f"{results['ff_A_global'][eid, 1]:+14.6e}  "
                  f"{results['ff_A_global'][eid, 2]:+14.6e}")
            print(f"  │  {'B':<5} "
                  f"{results['ff_B_global'][eid, 0]:+14.6e}  "
                  f"{results['ff_B_global'][eid, 1]:+14.6e}  "
                  f"{results['ff_B_global'][eid, 2]:+14.6e}")
            
            # ── Local end forces ──
            print(f"  │")
            print(f"  │  Local End Forces:")
            print(f"  │  {'End':<5} {'Fx_loc':>14}  "
                  f"{'Fz_loc':>14}  {'My_loc':>14}")
            print(f"  │  {'─'*52}")
            print(f"  │  {'A':<5} "
                  f"{results['ff_A_local'][eid, 0]:+14.6e}  "
                  f"{results['ff_A_local'][eid, 1]:+14.6e}  "
                  f"{results['ff_A_local'][eid, 2]:+14.6e}")
            print(f"  │  {'B':<5} "
                  f"{results['ff_B_local'][eid, 0]:+14.6e}  "
                  f"{results['ff_B_local'][eid, 1]:+14.6e}  "
                  f"{results['ff_B_local'][eid, 2]:+14.6e}")
            
            # ── Internal forces ──
            print(f"  │")
            print(f"  │  Internal Forces (beam convention):")
            print(f"  │  {'End':<5} {'N':>14}  {'V':>14}  {'M':>14}")
            print(f"  │  {'─'*52}")
            print(f"  │  {'A':<5} "
                  f"{results['N_A'][eid]:+14.6e}  "
                  f"{results['V_A'][eid]:+14.6e}  "
                  f"{results['M_A'][eid]:+14.6e}")
            print(f"  │  {'B':<5} "
                  f"{results['N_B'][eid]:+14.6e}  "
                  f"{results['V_B'][eid]:+14.6e}  "
                  f"{results['M_B'][eid]:+14.6e}")
            
            # ── Global end displacements ──
            print(f"  │")
            print(f"  │  Global End Displacements:")
            print(f"  │  {'End':<5} {'ux':>14}  {'uz':>14}  {'θy':>14}")
            print(f"  │  {'─'*52}")
            print(f"  │  {'A':<5} "
                  f"{results['disp_A_global'][eid, 0]:+14.6e}  "
                  f"{results['disp_A_global'][eid, 1]:+14.6e}  "
                  f"{results['disp_A_global'][eid, 2]:+14.6e}")
            print(f"  │  {'B':<5} "
                  f"{results['disp_B_global'][eid, 0]:+14.6e}  "
                  f"{results['disp_B_global'][eid, 1]:+14.6e}  "
                  f"{results['disp_B_global'][eid, 2]:+14.6e}")
            
            # ── Local end displacements ──
            print(f"  │")
            print(f"  │  Local End Displacements:")
            print(f"  │  {'End':<5} {'u_L':>14}  {'w_L':>14}  {'θ_L':>14}")
            print(f"  │  {'─'*52}")
            print(f"  │  {'A':<5} "
                  f"{results['disp_A_local'][eid, 0]:+14.6e}  "
                  f"{results['disp_A_local'][eid, 1]:+14.6e}  "
                  f"{results['disp_A_local'][eid, 2]:+14.6e}")
            print(f"  │  {'B':<5} "
                  f"{results['disp_B_local'][eid, 0]:+14.6e}  "
                  f"{results['disp_B_local'][eid, 1]:+14.6e}  "
                  f"{results['disp_B_local'][eid, 2]:+14.6e}")
            
            print(f"  └{'─'*80}")

    def print_reactions(self, results):
        """
        Print reaction forces at all support nodes.
        """
        print(f"\n{'═'*70}")
        print(f"  REACTION FORCES (Global)")
        print(f"{'═'*70}")
        
        sup_ids = results['support_node_ids']
        reactions = results['reactions_global']
        
        print(f"  {'Node':>6}  {'Rx':>14}  {'Rz':>14}  {'M_react':>14}  "
              f"{'Type':<20}")
        print(f"  {'─'*72}")
        
        total_Rx = 0.0
        total_Rz = 0.0
        total_M  = 0.0
        
        for i, nid in enumerate(sup_ids):
            bc_d = results['bc_disp'][nid, 0]
            bc_r = results['bc_rot'][nid, 0]
            if bc_d > 0.5 and bc_r > 0.5:
                stype = "Fixed"
            elif bc_d > 0.5:
                stype = "Pinned"
            else:
                stype = "Other"
            
            rx = reactions[i, 0]
            rz = reactions[i, 1]
            mr = reactions[i, 2]
            print(f"  {nid:6d}  {rx:+14.6e}  {rz:+14.6e}  "
                  f"{mr:+14.6e}  {stype:<20}")
            
            total_Rx += rx
            total_Rz += rz
            total_M  += mr
        
        print(f"  {'─'*72}")
        print(f"  {'Total':>6}  {total_Rx:+14.6e}  {total_Rz:+14.6e}  "
              f"{total_M:+14.6e}")
        print(f"{'═'*70}")

    # ════════════════════════════════════════════════════════
    # D. EXPORT TO CSV / DATAFRAME
    # ════════════════════════════════════════════════════════

    def to_node_dataframe(self, results, node_ids=None):
        """
        Create a pandas DataFrame with per-node results.
        
        Columns:
            node_id, x, y, z,
            ux, uz, theta_y,
            true_ux, true_uz, true_theta_y,  (if available)
            err_ux, err_uz, err_theta_y,     (if available)
            is_support, bc_disp, bc_rot,
            face0_Fx, face0_Fz, face0_My,
            face1_Fx, face1_Fz, face1_My,
            face2_Fx, face2_Fz, face2_My,
            face3_Fx, face3_Fz, face3_My,
            sum_Fx, sum_Fz, sum_My
        """
        if node_ids is None:
            node_ids = list(range(results['n_nodes']))
        
        rows = []
        face_names = ['+x', '-x', '+z', '-z']
        
        for nid in node_ids:
            row = {
                'node_id': nid,
                'x': results['coords'][nid, 0],
                'y': results['coords'][nid, 1],
                'z': results['coords'][nid, 2],
                'ux': results['disp_global'][nid, 0],
                'uz': results['disp_global'][nid, 1],
                'theta_y': results['disp_global'][nid, 2],
                'is_support': results['is_support'][nid],
                'bc_disp': results['bc_disp'][nid, 0],
                'bc_rot': results['bc_rot'][nid, 0],
            }
            
            if results['true_disp'] is not None:
                row['true_ux'] = results['true_disp'][nid, 0]
                row['true_uz'] = results['true_disp'][nid, 1]
                row['true_theta_y'] = results['true_disp'][nid, 2]
                row['err_ux'] = results['disp_error'][nid, 0]
                row['err_uz'] = results['disp_error'][nid, 1]
                row['err_theta_y'] = results['disp_error'][nid, 2]
            
            for f in range(4):
                fn = face_names[f].replace('+', 'p').replace('-', 'm')
                row[f'face_{fn}_Fx'] = results['face_forces_global'][nid, f, 0]
                row[f'face_{fn}_Fz'] = results['face_forces_global'][nid, f, 1]
                row[f'face_{fn}_My'] = results['face_forces_global'][nid, f, 2]
            
            row['sum_Fx'] = results['face_forces_global'][nid, :, 0].sum()
            row['sum_Fz'] = results['face_forces_global'][nid, :, 1].sum()
            row['sum_My'] = results['face_forces_global'][nid, :, 2].sum()
            
            rows.append(row)
        
        return pd.DataFrame(rows)

    def to_element_dataframe(self, results, element_ids=None):
        """
        Create a pandas DataFrame with per-element results.
        
        Columns:
            elem_id, nodeA, nodeB, length, angle_deg,
            E, A, I,
            ffA_Fx_glob, ffA_Fz_glob, ffA_My_glob,
            ffB_Fx_glob, ffB_Fz_glob, ffB_My_glob,
            ffA_Fx_loc,  ffA_Fz_loc,  ffA_My_loc,
            ffB_Fx_loc,  ffB_Fz_loc,  ffB_My_loc,
            N_A, V_A, M_A, N_B, V_B, M_B,
            dispA_ux, dispA_uz, dispA_thy,
            dispB_ux, dispB_uz, dispB_thy,
            dispA_uL, dispA_wL, dispA_thL,
            dispB_uL, dispB_wL, dispB_thL
        """
        if element_ids is None:
            element_ids = list(range(results['n_elements']))
        
        rows = []
        for eid in element_ids:
            d = results['elem_directions'][eid]
            angle = np.degrees(np.arctan2(d[2], d[0]))
            
            row = {
                'elem_id': eid,
                'nodeA': results['connectivity'][eid, 0],
                'nodeB': results['connectivity'][eid, 1],
                'length': results['elem_lengths'][eid],
                'angle_deg': angle,
                'E': results['prop_E'][eid],
                'A': results['prop_A'][eid],
                'I': results['prop_I'][eid],
                
                # Global end forces
                'ffA_Fx_glob': results['ff_A_global'][eid, 0],
                'ffA_Fz_glob': results['ff_A_global'][eid, 1],
                'ffA_My_glob': results['ff_A_global'][eid, 2],
                'ffB_Fx_glob': results['ff_B_global'][eid, 0],
                'ffB_Fz_glob': results['ff_B_global'][eid, 1],
                'ffB_My_glob': results['ff_B_global'][eid, 2],
                
                # Local end forces
                'ffA_Fx_loc': results['ff_A_local'][eid, 0],
                'ffA_Fz_loc': results['ff_A_local'][eid, 1],
                'ffA_My_loc': results['ff_A_local'][eid, 2],
                'ffB_Fx_loc': results['ff_B_local'][eid, 0],
                'ffB_Fz_loc': results['ff_B_local'][eid, 1],
                'ffB_My_loc': results['ff_B_local'][eid, 2],
                
                # Internal forces
                'N_A': results['N_A'][eid],
                'V_A': results['V_A'][eid],
                'M_A': results['M_A'][eid],
                'N_B': results['N_B'][eid],
                'V_B': results['V_B'][eid],
                'M_B': results['M_B'][eid],
                
                # Global end displacements
                'dispA_ux': results['disp_A_global'][eid, 0],
                'dispA_uz': results['disp_A_global'][eid, 1],
                'dispA_thy': results['disp_A_global'][eid, 2],
                'dispB_ux': results['disp_B_global'][eid, 0],
                'dispB_uz': results['disp_B_global'][eid, 1],
                'dispB_thy': results['disp_B_global'][eid, 2],
                
                # Local end displacements
                'dispA_uL': results['disp_A_local'][eid, 0],
                'dispA_wL': results['disp_A_local'][eid, 1],
                'dispA_thL': results['disp_A_local'][eid, 2],
                'dispB_uL': results['disp_B_local'][eid, 0],
                'dispB_wL': results['disp_B_local'][eid, 1],
                'dispB_thL': results['disp_B_local'][eid, 2],
            }
            rows.append(row)
        
        return pd.DataFrame(rows)

    def export_csv(self, results, output_dir="RESULTS",
                   node_ids=None, element_ids=None):
        """
        Export results to CSV files.
        
        Creates:
            output_dir/node_results.csv
            output_dir/element_results.csv
            output_dir/reactions.csv
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # ── Node results ──
        df_nodes = self.to_node_dataframe(results, node_ids)
        node_path = os.path.join(output_dir, 'node_results.csv')
        df_nodes.to_csv(node_path, index=False, float_format='%.10e')
        print(f"  ✓ Node results saved: {node_path} "
              f"({len(df_nodes)} nodes)")
        
        # ── Element results ──
        df_elems = self.to_element_dataframe(results, element_ids)
        elem_path = os.path.join(output_dir, 'element_results.csv')
        df_elems.to_csv(elem_path, index=False, float_format='%.10e')
        print(f"  ✓ Element results saved: {elem_path} "
              f"({len(df_elems)} elements)")
        
        # ── Reactions ──
        sup_ids = results['support_node_ids']
        reactions = results['reactions_global']
        df_react = pd.DataFrame({
            'node_id': sup_ids,
            'Rx': reactions[:, 0],
            'Rz': reactions[:, 1],
            'M_react': reactions[:, 2],
        })
        react_path = os.path.join(output_dir, 'reactions.csv')
        df_react.to_csv(react_path, index=False, float_format='%.10e')
        print(f"  ✓ Reactions saved: {react_path} "
              f"({len(df_react)} supports)")
        
        return df_nodes, df_elems, df_react

    # ════════════════════════════════════════════════════════
    # E. SUMMARY TABLE
    # ════════════════════════════════════════════════════════

    def print_summary(self, results):
        """
        Print a compact summary table of all node results.
        """
        print(f"\n{'═'*110}")
        print(f"  SUMMARY: Global Displacements at All Nodes")
        print(f"{'═'*110}")
        print(f"  {'Node':>4}  {'x':>8}  {'z':>8}  │  "
              f"{'ux':>14}  {'uz':>14}  {'θy':>14}  │  "
              f"{'Type':<15}")
        print(f"  {'─'*108}")
        
        for nid in range(results['n_nodes']):
            x = results['coords'][nid, 0]
            z = results['coords'][nid, 2]
            ux = results['disp_global'][nid, 0]
            uz = results['disp_global'][nid, 1]
            th = results['disp_global'][nid, 2]
            
            bc_d = results['bc_disp'][nid, 0]
            bc_r = results['bc_rot'][nid, 0]
            if bc_d > 0.5 and bc_r > 0.5:
                ntype = "Fixed"
            elif bc_d > 0.5:
                ntype = "Pinned"
            else:
                ntype = "Free"
            
            print(f"  {nid:4d}  {x:8.3f}  {z:8.3f}  │  "
                  f"{ux:+14.6e}  {uz:+14.6e}  {th:+14.6e}  │  "
                  f"{ntype:<15}")
        
        print(f"  {'─'*108}")
        
        # ── Error summary ──
        if results['true_disp'] is not None:
            err = results['disp_error']
            print(f"\n  Displacement Error Summary:")
            print(f"    Max |err_ux|: {np.abs(err[:, 0]).max():.6e}")
            print(f"    Max |err_uz|: {np.abs(err[:, 1]).max():.6e}")
            print(f"    Max |err_θy|: {np.abs(err[:, 2]).max():.6e}")
            print(f"    RMS error:    {np.sqrt((err**2).mean()):.6e}")
            rel_err = np.sqrt((err**2).sum()) / max(
                np.sqrt((results['true_disp']**2).sum()), 1e-10)
            print(f"    Relative L2:  {rel_err:.6e}")


# ════════════════════════════════════════════════════════
# MAIN — Run post-processing
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    
    print("=" * 60)
    print("  PIGNN POST-PROCESSING")
    print("=" * 60)
    
    # ── Configuration ──
    CHECKPOINT = "RESULTS/best.pt"
    DATA_PATH  = "DATA/graph_dataset.pt"
    OUTPUT_DIR = "RESULTS"
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT):
        print(f"\n  ⚠ Checkpoint not found: {CHECKPOINT}")
        print(f"  Run train.py first, or specify a different path.")
        print(f"  Trying 'RESULTS/final.pt' instead...")
        CHECKPOINT = "RESULTS/final.pt"
    
    if not os.path.exists(CHECKPOINT):
        print(f"  ⚠ No checkpoint found. Please train the model first.")
        exit(1)
    
    # ── Create post-processor ──
    pp = PIGNNPostProcessor(
        checkpoint_path=CHECKPOINT,
        data_path=DATA_PATH,
        hidden_dim=128,
        n_layers=6,
        node_in_dim=9,
        edge_in_dim=10,
    )
    
    # ── Extract results for graph 0 ──
    results = pp.extract_all(graph_idx=0)
    
    # ── Print summary ──
    pp.print_summary(results)
    
    # ── Print detailed node results ──
    # Specify node IDs or None for all
    # Example: first 5 nodes + all support nodes
    all_node_ids = list(range(min(5, results['n_nodes'])))
    all_node_ids.extend(results['support_node_ids'])
    all_node_ids = sorted(set(all_node_ids))
    pp.print_node_results(results, node_ids=all_node_ids)
    
    # ── Print element results ──
    # First 3 elements
    pp.print_element_results(
        results,
        element_ids=list(range(min(3, results['n_elements'])))
    )
    
    # ── Print reactions ──
    pp.print_reactions(results)
    
    # ── Export to CSV ──
    df_nodes, df_elems, df_react = pp.export_csv(
        results, output_dir=OUTPUT_DIR
    )
    
    # ── Show DataFrame preview ──
    print(f"\n  Node DataFrame preview:")
    print(df_nodes[['node_id', 'x', 'z', 'ux', 'uz', 'theta_y',
                     'is_support']].to_string(index=False))
    
    print(f"\n  Element DataFrame preview (first 5):")
    cols = ['elem_id', 'nodeA', 'nodeB', 'angle_deg',
            'N_A', 'V_A', 'M_A', 'N_B', 'V_B', 'M_B']
    print(df_elems[cols].head().to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"  POST-PROCESSING COMPLETE ✓")
    print(f"  Output files in {OUTPUT_DIR}/:")
    print(f"    node_results.csv")
    print(f"    element_results.csv")
    print(f"    reactions.csv")
    print(f"{'='*60}")