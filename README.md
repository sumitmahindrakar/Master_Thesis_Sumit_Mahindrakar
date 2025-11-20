# Master_Thesis_Sumit_Mahindrakar

This project uses Kratos Multiphysics.

## Setup

Kratos must be installed and built first. Follow steps below.

1. Clone Kratos:

```bash
mkdir -p ~/software
cd ~/software
git clone https://github.com/KratosMultiphysics/Kratos.git
```

2. Clone applications:

```bash
cd ~/software/Kratos
git clone https://github.com/KratosMultiphysics/StructuralMechanicsApplication.git applications/StructuralMechanicsApplication
git clone https://github.com/KratosMultiphysics/LinearSolversApplication.git applications/LinearSolversApplication
```

3. Build Kratos:

```bash
cd ~/software/Kratos
mkdir build && cd build
cmake .. -DPython_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

4. Set PYTHONPATH permanently:

```bash
echo "export PYTHONPATH=~/software/Kratos/bin/Release:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

5. Run your scripts from the `scripts/` folder:

```bash
python scripts/main.py
```

