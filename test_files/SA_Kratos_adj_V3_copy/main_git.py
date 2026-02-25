import os
import time
import subprocess

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3_copy")
print(f"Working directory: {os.getcwd()}")

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.KratosUnittest as KratosUnittest
from KratosMultiphysics.StructuralMechanicsApplication import structural_mechanics_analysis
import KratosMultiphysics.kratos_utilities as kratos_utilities
from KratosMultiphysics import IsDistributedRun
from structural_mechanics_test_factory import SelectAndVerifyLinearSolver

has_hdf5_application = kratos_utilities.CheckIfApplicationsAvailable("HDF5Application")


# ============================================================
# CHANGE 1: Add this class
# ============================================================
class CustomPrimalAnalysis(structural_mechanics_analysis.StructuralMechanicsAnalysis):
    """Custom analysis that transfers condition loads to nodes for VTK output."""
    
    def FinalizeSolutionStep(self):
        """Transfer loads AFTER they are applied, BEFORE VTK writes."""
        
        mp = self.model.GetModelPart("Structure")
        LINE_LOAD = StructuralMechanicsApplication.LINE_LOAD
        POINT_LOAD = StructuralMechanicsApplication.POINT_LOAD
        POINT_MOMENT = StructuralMechanicsApplication.POINT_MOMENT  # Added POINT_MOMENT
        
        # Initialize all nodes to zero
        zero = KratosMultiphysics.Array3([0.0, 0.0, 0.0])
        for node in mp.Nodes:
            node.SetValue(LINE_LOAD, zero)
            node.SetValue(POINT_LOAD, zero)
            node.SetValue(POINT_MOMENT, zero)  # Initialize POINT_MOMENT
        
        # Transfer LINE_LOAD from conditions to nodes
        try:
            load_mp = mp.GetSubModelPart("LineLoad3D_load")
            for condition in load_mp.Conditions:
                load_val = condition.GetValue(LINE_LOAD)
                for node in condition.GetGeometry():
                    node.SetValue(LINE_LOAD, load_val)
            print(f"Transferred LINE_LOAD to {load_mp.NumberOfNodes()} nodes")
        except:
            pass
        
        # Transfer POINT_LOAD from conditions to nodes
        try:
            load_mp = mp.GetSubModelPart("PointLoad3D_load")
            for condition in load_mp.Conditions:
                load_val = condition.GetValue(POINT_LOAD)
                for node in condition.GetGeometry():
                    node.SetValue(POINT_LOAD, load_val)
            print(f"Transferred POINT_LOAD to {load_mp.NumberOfNodes()} nodes")
        except:
            pass
        
        # Transfer POINT_MOMENT from conditions to nodes
        # Try different possible sub-model part names
        moment_submodelpart_names = [
            "PointMoment3D_moment",
            "PointMoment3D_load", 
            "PointMoment3D",
            "Moment3D_moment",
            "PointMoment"
        ]
        
        moment_transferred = False
        for submp_name in moment_submodelpart_names:
            try:
                moment_mp = mp.GetSubModelPart(submp_name)
                for condition in moment_mp.Conditions:
                    moment_val = condition.GetValue(POINT_MOMENT)
                    for node in condition.GetGeometry():
                        node.SetValue(POINT_MOMENT, moment_val)
                print(f"Transferred POINT_MOMENT to {moment_mp.NumberOfNodes()} nodes (from {submp_name})")
                moment_transferred = True
                break
            except:
                continue
        
        # Alternative: Search all conditions for POINT_MOMENT if no specific submodelpart found
        if not moment_transferred:
            try:
                moment_count = 0
                for condition in mp.Conditions:
                    if condition.Has(POINT_MOMENT):
                        moment_val = condition.GetValue(POINT_MOMENT)
                        # Check if it's non-zero
                        if abs(moment_val[0]) > 1e-12 or abs(moment_val[1]) > 1e-12 or abs(moment_val[2]) > 1e-12:
                            for node in condition.GetGeometry():
                                node.SetValue(POINT_MOMENT, moment_val)
                                moment_count += 1
                if moment_count > 0:
                    print(f"Transferred POINT_MOMENT to {moment_count} nodes (from all conditions)")
            except Exception as e:
                print(f"Note: No POINT_MOMENT conditions found or error: {e}")
        
        # Call parent AFTER transfer (this triggers VTK output)
        super().FinalizeSolutionStep()
# ============================================================


class AdjointSensitivityAnalysisTestFactory(KratosUnittest.TestCase):
    def setUp(self):
        with KratosUnittest.WorkFolderScope(".", __file__):
            with open(self.primal_file_name, 'r') as parameter_file:
                primal_parameters = KratosMultiphysics.Parameters(parameter_file.read())
            with open(self.adjoint_file_name, 'r') as parameter_file:
                self.adjoint_parameters = KratosMultiphysics.Parameters(parameter_file.read())
            self.problem_name = primal_parameters["problem_data"]["problem_name"].GetString()
            self.model_part_name = primal_parameters["solver_settings"]["model_part_name"].GetString()

            if (primal_parameters["problem_data"]["echo_level"].GetInt() == 0 or
                self.adjoint_parameters["problem_data"]["echo_level"].GetInt() == 0):
                KratosMultiphysics.Logger.GetDefaultOutput().SetSeverity(
                    KratosMultiphysics.Logger.Severity.WARNING)

            SelectAndVerifyLinearSolver(primal_parameters, self.skipTest)
            SelectAndVerifyLinearSolver(self.adjoint_parameters, self.skipTest)

            model_primal = KratosMultiphysics.Model()

            # ============================================================
            # CHANGE 2: Use CustomPrimalAnalysis instead of default
            # ============================================================
            primal_analysis = CustomPrimalAnalysis(
                model_primal, primal_parameters)

            # Initialize
            primal_analysis.Initialize()

            # Get model part
            _mp = model_primal.GetModelPart(self.model_part_name)

            # Transfer material properties to elements
            _vars = [
                KratosMultiphysics.YOUNG_MODULUS,
                KratosMultiphysics.DENSITY,
                KratosMultiphysics.POISSON_RATIO,
                StructuralMechanicsApplication.CROSS_AREA,
                StructuralMechanicsApplication.TORSIONAL_INERTIA,
                StructuralMechanicsApplication.I22,
                StructuralMechanicsApplication.I33,
            ]
            for element in _mp.Elements:
                props = element.Properties
                for var in _vars:
                    if props.Has(var):
                        element.SetValue(var, props[var])

            # NO manual load transfer needed — CustomPrimalAnalysis handles it!

            # Run solution
            primal_analysis.RunSolutionLoop()
            primal_analysis.Finalize()

            # Adjoint setup
            model_adjoint = KratosMultiphysics.Model()
            self.adjoint_analysis = structural_mechanics_analysis.StructuralMechanicsAnalysis(
                model_adjoint, self.adjoint_parameters)
            self.adjoint_analysis.Initialize()

    def test_execution(self):
        with KratosUnittest.WorkFolderScope(".", __file__):
            self.adjoint_analysis.RunSolutionLoop()
            self.perform_additional_checks()

    def perform_additional_checks(self):
        pass

    def tearDown(self):
        with KratosUnittest.WorkFolderScope(".", __file__):
            self.adjoint_analysis.Finalize()
            kratos_utilities.DeleteFileIfExisting(self.problem_name + ".time")
            kratos_utilities.DeleteFileIfExisting(self.model_part_name + ".h5")
            kratos_utilities.DeleteFileIfExisting(self.model_part_name + "-1.0000.h5")
            kratos_utilities.DeleteFileIfExisting(self.model_part_name + "-1.1000.h5")


@KratosUnittest.skipUnless(has_hdf5_application, "Missing required application: HDF5Application")
class TestAdjointSensitivityAnalysisBeamStructureLocalStress(AdjointSensitivityAnalysisTestFactory):
    primal_file_name = "beam_test_parameters.json"
    adjoint_file_name = "beam_test_local_stress_adjoint_parameters.json"

    def perform_additional_checks(self):
        element_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        adjoint_model_part = self.adjoint_analysis.model.GetModelPart(self.model_part_name)

        print("\n=== I22_SENSITIVITY (dMY/dI22) ===")
        for element_id in element_list:
            val = adjoint_model_part.Elements[element_id].GetValue(
                StructuralMechanicsApplication.I22_SENSITIVITY
            )
            print(f"  Element {element_id}: {val:.6f}")


if __name__ == '__main__':
    # start = time.time()

    suites = KratosUnittest.KratosSuites
    smallSuite = suites['small']
    smallSuite.addTests(KratosUnittest.TestLoader().loadTestsFromTestCases(
        [TestAdjointSensitivityAnalysisBeamStructureLocalStress]))
    allSuite = suites['all']
    allSuite.addTests(smallSuite)
    KratosUnittest.runTests(suites)
    
    
    # elapsed = time.time() - start
    # print(f"\nOne Kratos run: {elapsed:.1f} seconds")
    # print(f"500 cases would take: {elapsed * 500 / 60:.1f} minutes")
    # print(f"1000 cases would take: {elapsed * 1000 / 60:.1f} minutes")