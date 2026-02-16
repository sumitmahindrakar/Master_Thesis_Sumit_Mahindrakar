import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")
print(f"Working directory: {os.getcwd()}")

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.KratosUnittest as KratosUnittest
from KratosMultiphysics.StructuralMechanicsApplication import structural_mechanics_analysis
import KratosMultiphysics.kratos_utilities as kratos_utilities
from KratosMultiphysics import IsDistributedRun
from structural_mechanics_test_factory import SelectAndVerifyLinearSolver

has_hdf5_application = kratos_utilities.CheckIfApplicationsAvailable("HDF5Application")


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
            primal_analysis = structural_mechanics_analysis.StructuralMechanicsAnalysis(
                model_primal, primal_parameters)
            primal_analysis.Run()

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
    suites = KratosUnittest.KratosSuites
    smallSuite = suites['small']
    smallSuite.addTests(KratosUnittest.TestLoader().loadTestsFromTestCases(
        [TestAdjointSensitivityAnalysisBeamStructureLocalStress]))
    allSuite = suites['all']
    allSuite.addTests(smallSuite)
    KratosUnittest.runTests(suites)