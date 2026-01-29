el_part.CreateNewMasterSlaveConstraint(
            "LinearMasterSlaveConstraint", 3,
            node_left, KratosMultiphysics.ROTATION_Z,
            node_right, KratosMultiphysics.ROTATION_Z,
            1.0, 1.0 # slave = 1.0 * master + 1.0
        )