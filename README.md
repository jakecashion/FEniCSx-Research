# FEniCSx Research

Finite element simulations for electromagnetic problems using [FEniCSx](https://fenicsproject.org/).

## Main Research

- **`maxwell.py`** — Maxwell's equations with impedance boundary damping on a subset of the boundary and PEC (perfect electric conductor) conditions elsewhere. Solves on a 3D unit cube using Nedelec curl-conforming elements. Outputs the electric field solution to XDMF for visualization in ParaView.

  ```bash
  mpirun -n 4 python3 maxwell.py
  ```

## Project Structure

```
├── maxwell.py                          # Main research solver
├── electromagnetics/
│   └── electromagnetism.py             # 2D magnetostatics (iron cylinders + copper coils)
├── fundamentals/
│   └── poisson_equation.py             # Poisson equation tutorial example
├── reference/
│   ├── demo_pml.py                     # FEniCSx PML scattering demo
│   ├── maxwell_pml_demo.py             # PML demo (combined mesh + solver)
│   ├── cavity_eigenvalue.py            # SLEPc cavity resonance eigenvalue problem
│   ├── run_cavity.py                   # Combined cavity mesh generation + eigenvalue solve
│   └── generate_mesh.py               # Standalone gmsh mesh generation
├── test_fenicsx.py                     # Quick installation verification
├── environment.yml                     # Conda environment specification
└── .gitignore
```

## Setup (Conda)

1. Install [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Create and activate the environment:

   ```bash
   conda env create -f environment.yml
   conda activate fenicsx
   ```

3. Verify the installation:

   ```bash
   python test_fenicsx.py
   ```

**Note:** `maxwell.py` requires DOLFINx compiled with complex PETSc scalar support. The `environment.yml` pins `petsc=*=*complex*` to ensure this.

## Visualization

Output files (`.xdmf`, `.h5`) can be opened with [ParaView](https://www.paraview.org/) for post-processing and visualization.
