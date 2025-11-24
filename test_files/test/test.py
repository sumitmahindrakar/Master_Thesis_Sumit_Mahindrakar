import KratosMultiphysics as KM

print("\n============================")
print("   SCALAR (Double) variables")
print("============================")
for name in list(KM.DoubleVariables.keys()):
    print(name)
print("Total:", len(KM.DoubleVariables))


print("\n============================")
print("      ARRAY_3 variables")
print("============================")
for name in list(KM.Array1DVariables.keys()):
    print(name)
print("Total:", len(KM.Array1DVariables))


print("\n============================")
print("      INTEGER variables")
print("============================")
for name in list(KM.IntegerVariables.keys()):
    print(name)
print("Total:", len(KM.IntegerVariables))


print("\n============================")
print("      BOOL variables")
print("============================")
for name in list(KM.BoolVariables.keys()):
    print(name)
print("Total:", len(KM.BoolVariables))


print("\n============================")
print("      VECTOR variables")
print("============================")
for name in list(KM.VectorVariables.keys()):
    print(name)
print("Total:", len(KM.VectorVariables))
