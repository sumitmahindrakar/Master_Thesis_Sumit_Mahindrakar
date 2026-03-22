# 5. Shape function forces (REPLACES autograd)          # ◀◀◀ CHANGED
        N_sf, M_A_sf, M_B_sf, V_sf = \                         # ◀◀◀ NEW
            self._compute_shape_function_forces(                 # ◀◀◀ NEW
                disp_A_loc, disp_B_loc, data                     # ◀◀◀ NEW
            )                                                    # ◀◀◀ NEW
