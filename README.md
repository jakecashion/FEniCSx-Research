# 📚 FEniCSx Research Archive

This repository serves as a centralized workspace for Jake Cashion's research using [FEniCSx](https://fenicsproject.org/fenicsx/), focused on electromagnetic scattering, symbolic modeling, and modular simulation workflows.

It includes:
- Simulation scripts and demos (e.g., `demo_pml.py`)
- Mesh generation and field output files
- Docker launch templates for reproducible environments
- Visualization-ready data for ParaView and PyVista
- Notes, experiments, and symbolic modeling drafts

---

## 🚀 Running the Demo in Docker (Complex Mode)

To run the electromagnetic scattering demo inside a Docker container:

```bash
docker run -ti \
  -v /path/to/local/folder:/home/fenics/shared \
  -w /home/fenics/shared \
  dolfinx/dolfinx:stable \
  bash -c "source /usr/local/bin/dolfinx-complex-mode && bash"
