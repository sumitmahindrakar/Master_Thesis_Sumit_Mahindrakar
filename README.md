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




# Sensitivity Analysis Pipeline for Beam and Frame Structures

A complete automated pipeline for computing bending moment sensitivity (∂M/∂EI) 
using the adjoint method with Kratos Multiphysics.

## Overview

This pipeline automates the entire workflow for structural sensitivity analysis:

1. **Mesh Refinement**: Refines coarse template mesh to desired resolution
2. **Dual MDPA Generation**: Automatically creates dual analysis mesh with kink at response location
3. **Primary Analysis**: Runs structural analysis with applied loads
4. **Dual Analysis**: Runs adjoint analysis with unit kink
5. **Sensitivity Computation**: Computes ∂M/∂(EI) for all elements
6. **Visualization**: Generates publication-ready plots

## Quick Start

### 1. Edit Configuration

Edit `config.yaml` to set your problem:

```yaml
problem:
  template: "beam"           # Options: beam, frame_1story, frame_2story
  name: "my_beam_analysis"

mesh:
  subdivisions: 10           # Elements per original element

response:
  x: 1.0                     # X-coordinate for sensitivity response
  y: 0.0                     # Y-coordinate for sensitivity response
