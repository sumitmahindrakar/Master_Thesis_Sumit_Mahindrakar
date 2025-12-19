import sys
import time
import importlib

import KratosMultiphysics

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model,project_parameters, flush_frequency=10.0):
            super().__init__(model,project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()

            if hasattr(self, "_hinge_created"):
                return
            self._hinge_created = True

            model_part = self.model["Structure"]

            node_2 = model_part.GetNode(2)
            node_6 = model_part.GetNode(6)

            cid = model_part.NumberOfMasterSlaveConstraints() + 1

            model_part.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint",
                cid,
                node_2,
                KratosMultiphysics.DISPLACEMENT_X,
                node_6,
                KratosMultiphysics.DISPLACEMENT_X,
                1.0,
                0.0
            )

            cid += 1
            model_part.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint",
                cid,
                node_2,
                KratosMultiphysics.DISPLACEMENT_Y,
                node_6,
                KratosMultiphysics.DISPLACEMENT_Y,
                1.0,
                0.0
            )

            
            # print("Hinge constraint created: Node 4 follows Node 2 displacement")
            # print("Rotation is FREE (hinge behavior)")
            # ================================================
            
            sys.stdout.flush()

            sys.stdout.flush()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    with open("test_files/SA_beam_2D_udl_kink.gid/ProjectParameters_dual.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()
