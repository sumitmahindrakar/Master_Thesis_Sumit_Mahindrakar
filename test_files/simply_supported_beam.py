"""
Simply Supported Beam with fixed end and loaded at middle

Length=10 m
Youngs modulud=206.9 GPa
Poisson ratio=0.29
Load at node 6=10 N
                                                 |
                                                 |
1──(1)───2───(2)──3──(3)───4───(4)──5───(5)──6──(6)───7───(7)──8──(8)───9──(9)───10──(10)───11
/\                                                                                          /\ 
    
"""

import KratosMultiphysics as Kratos #Kratos does not have properties therefore following import
import KratosMultiphysics.StructuralMechanicsApplication as KratosSA
from KratosMultiphysics import python_linear_solver_factory as linear_solver_factory

# create Germetry or mesh
def CreateModel() -> Kratos.Model: #def returns a model
    model= Kratos.Model()
    structure = model.CreateModelPart("Structure")
    structure.SetBufferSize(2) # can recall 2 previous time steps
    # this is a 1D problem
    structure.ProcessInfo[Kratos.DOMAIN_SIZE] = 2
    structure.CloneTimeStep(1.0)

    # define storage allocation
    # allocation of variables - so it always put hystorical date first so it si easy to work on
    structure.AddNodalSolutionStepVariable(Kratos.DISPLACEMENT)
    structure.AddNodalSolutionStepVariable(Kratos.REACTION)
    # we are looking at these values above mentioned

    # Now add nodes********************

    structure.CreateNewNode(1,0,0,0)# in Kratos index always starts with 1 not 0 and it is print(node number then comes (X,Y,Z) coordinates
    structure.CreateNewNode(2,1,0,0)
    structure.CreateNewNode(3,2,0,0)
    structure.CreateNewNode(4,3,0,0)
    structure.CreateNewNode(5,4,0,0)
    structure.CreateNewNode(6,5,0,0)
    structure.CreateNewNode(7,6,0,0)
    structure.CreateNewNode(8,7,0,0)
    structure.CreateNewNode(9,8,0,0)
    structure.CreateNewNode(10,9,0,0)
    structure.CreateNewNode(11,10,0,0)

    # node: Kratos.Node = structure.CreateNewNode(12,9,2,0)

    # node.SetSolutionStepValue(Kratos.DISPLACEMENT_X, 1.5) #set the displacement values of X
    # node.SetSolutionStepValue(Kratos.DISPLACEMENT_X, 1, 1.5) #set the displacement values of X for previous time step by 1(2 set the second previous time step)
    # node.SetValue(Kratos.DISPLACEMENT, Kratos.Array3([1,2,3])) #set current values, Displ is first Order tensor so Arr3
    
    # node.SetValue(Kratos.VELOCITY, Kratos.Array3([1,2,3]))

    # print(node.GetSolutionStepValue(Kratos.DISPLACEMENT_X, 1))#get the displacement values of X coord from previ time Step
    # print(node.GetSolutionStepValue(Kratos.DISPLACEMENT))#get the displacement values of all coord
    # print(node.GetValue(Kratos.VELOCITY))

    #Flages
    # print(node.Is(Kratos.ACTIVE))

    #Has
    # print(node.Has(Kratos.VELOCITY))
    # print(node.Has(Kratos.ACCELERATION))
    # print(node.HasSolutionStepValue(Kratos.VELOCITY)) #check hystorical values



    #create Properties
    properties= structure.CreateNewProperties(1)

    #Create Domain element********************
    solid = structure.CreateSubModelPart("solid")# subsub for next subsub model part
    solid.AddNodes([1,2,3,4,5,6,7,8,9,10,11]) #allocate the nodes to sub model from model (parent model)

    #add element to this solid
    solid.CreateNewElement("LineElement2D2N", 1 , [1,2], properties)# 1 is ID or index, SmallDisplacementElement2D2N
    # element: Kratos.Element = solid.CreateNewElement("SmallDisplacementElement2D4N", 1 , [1,2,6,5], properties)
    solid.CreateNewElement("LineElement2D2N", 2 , [2,3], properties)
    solid.CreateNewElement("LineElement2D2N", 3 , [3,4], properties)
    solid.CreateNewElement("LineElement2D2N", 4 , [4,5], properties)
    solid.CreateNewElement("LineElement2D2N", 5 , [5,6], properties)
    solid.CreateNewElement("LineElement2D2N", 6 , [6,7], properties)
    solid.CreateNewElement("LineElement2D2N", 7 , [7,8], properties)
    solid.CreateNewElement("LineElement2D2N", 8 , [8,9], properties)
    solid.CreateNewElement("LineElement2D2N", 9 , [9,10], properties)
    solid.CreateNewElement("LineElement2D2N", 10 , [10,11], properties)

    #adding Loads********************
    load = structure.CreateSubModelPart("load")
    load.AddNodes([6])

    load_properties= structure.CreateNewProperties(2)
    load.CreateNewCondition("PointLoadCondition2D1N", 1, [6], load_properties)

    
    #adding fixture node
    fixture=structure.CreateSubModelPart("fixture")
    fixture.AddNodes([1,11])

    #Dirichlet BC (General BC)
    node: Kratos.Node
    for node in structure.Nodes:
        node.AddDof(Kratos.DISPLACEMENT_X)
        node.AddDof(Kratos.DISPLACEMENT_Y)
        node.AddDof(Kratos.DISPLACEMENT_Z)
        # print(node)

    print(structure)
    return model
    


#apply material properties********************
def ApplyMaterialProperties(structure:Kratos.ModelPart) -> None:
    properties = structure.GetProperties(1)# for structure(solid) ID is 1 already assigned
    cl = Kratos.KratosGlobals.GetConstitutiveLaw("LinearElasticPlaneStrain2DLaw").Clone()
    properties.SetValue(Kratos.CONSTITUTIVE_LAW,cl)
    # properties[Kratos.CONSTITUTIVE_LAW]=cl
    properties[Kratos.YOUNG_MODULUS]=206.9e9
    properties[Kratos.POISSON_RATIO]=0.29

    # Solve this issue
    properties[Kratos.MOMENT_X]=1


#apply BCs ********************
def ApplyBoundaryConditions(model:Kratos.Model) -> None:
    #apply DBC
    node:Kratos.Node
    for node in model.GetModelPart("Structure.fixture").Nodes:
        node.Fix(Kratos.DISPLACEMENT_X)
        node.Fix(Kratos.DISPLACEMENT_Y)
        node.Fix(Kratos.DISPLACEMENT_Z)
        # node.FREE(Kratos.DISPLACEMENT_Z)

    #apply loads
    condition:Kratos.Condition
    for condition in model.GetModelPart("Structure.load").Conditions:
        condition.SetValue(KratosSA.LINE_LOAD, Kratos.Array3([0,-10,0]))


#Solver
# def Solve(model: Kratos.Model) -> None:
#     # linear_solver = Kratos.LinearSolverFactory().Create(Kratos.Parameters("""{"solver_type":"amgcl"}"""))
    
#     linear_solver = linear_solver_factory.CreateFastestAvailableDirectLinearSolver()
#     builder_and_solver = Kratos.ResidualBasedBlockBuilderAndSolver(linear_solver)
#     convergence_criterion= Kratos.ResidualCriteria(1e-6,1e-9)#rel tol, abs tol
#     scheme = KratosSA.StructuralMechanicsStaticScheme(Kratos.Parameters("""{}"""))
    
    
    
#     strategy = Kratos.ResidualBasedNewtonRaphsonStrategy(model.GetModelPart("Structure"),
                                                                    #  scheme,
                                                                    #  convergence_criterion,
                                                                    #  builder_and_solver,
                                                                    #  100,#max iter,
                                                                    #  False, #Compute reaction,
                                                                    #  False, # no remashing .... self.settings["reform_dofs_at_each_step"].GetBool(),
                                                                    #  False, # no move mesh ..... self.settings["move_mesh_flag"].GetBool()
                                                                    # )
    # strategy.Solve()

if __name__ == "__main__":
    model = CreateModel()
    structure = model.GetModelPart("Structure")
    ApplyMaterialProperties(structure)
    ApplyBoundaryConditions(model)
    # Solve(model)

    scheme = KratosSA.StructuralMechanicsStaticScheme(Kratos.Parameters("""{}"""))
    linear_solver = linear_solver_factory.CreateFastestAvailableDirectLinearSolver()
    convergence_criteria = KratosSA.ResidualDisplacementAndOtherDoFCriteria(1e-6, 1e-9)
    builder_and_solver = Kratos.ResidualBasedBlockBuilderAndSolver(linear_solver)

    max_iterations = 10
    compute_reactions = False
    reform_dofs_at_each_step = False
    move_mesh_flag = False

    strategy = Kratos.ResidualBasedNewtonRaphsonStrategy(model["Structure"], scheme, convergence_criteria, builder_and_solver, max_iterations, compute_reactions, reform_dofs_at_each_step, move_mesh_flag)
    strategy.Solve()
    
    #output
    # vtu_output = Kratos.VtuOutput(model["Structure"])
    # vtu_output.AddHistoricalVariable(Kratos.DISPLACEMENT)
    # vtu_output.PrintOutput("test_files/simplt_supported_beam")

    # for element in structure.Elements:
    #     print(element.Properties[Kratos.YOUNG_MODULUS])

    # print("Displacement - ")
    # node: Kratos.Node = structure.CreateNewNode(6,5,0,0)
    # print(node.GetSolutionStepValue(Kratos.DISPLACEMENT))#get the displacement values of all coord
    # # print(node.GetValue(Kratos.DISPLACEMENT))
    # print("end")

    