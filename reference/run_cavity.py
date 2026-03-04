import gmsh
import json
import ufl
import numpy as np
import sys
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import fem, io, mesh
from dolfinx.mesh import gmshio

# ====================================================================
# PART 1: MESH GENERATION (from generate_mesh.py)
# ====================================================================

gmsh.initialize(sys.argv)
model = gmsh.model
model.add("cavity")

# Parameters
L = 1.0
c_x = 0.0
c_y = 0.0
c_z = 0.0
r_ext = 0.3
r_int = 0.1
l_ext = 1.0
l_int = 0.4

# Create geometry
box = model.occ.addBox(c_x - L / 2, c_y - L / 2, c_z - L / 2, L, L, L)
cyl_ext = model.occ.addCylinder(c_x, c_y, c_z - l_ext / 2, 0, 0, l_ext, r_ext)
cyl_int = model.occ.addCylinder(c_x, c_y, c_z - l_int / 2, 0, 0, l_int, r_int)
cut = model.occ.cut([(3, cyl_ext)], [(3, cyl_int)], removeTool=True)
fusion = model.occ.fuse([(3, box)], cut[0])
model.occ.synchronize()

# Add physical groups and name them
cavity_marker = 1
boundary_marker = 2

volumes = model.getEntities(3)
model.addPhysicalGroup(3, [volumes[0][1]], cavity_marker)
model.setPhysicalName(3, cavity_marker, "cavity")

surfaces = model.occ.getEntities(2)
boundary_surfaces = []
for surface in surfaces:
    com = model.occ.getCenterOfMass(surface[0], surface[1])
    if not np.allclose(com, [0, 0, 0]):
        boundary_surfaces.append(surface[1])
model.addPhysicalGroup(2, boundary_surfaces, boundary_marker)
model.setPhysicalName(2, boundary_marker, "boundary")

# Set mesh size and generate mesh
lc = 0.1
model.mesh.setSize(model.getEntities(0), lc)
model.mesh.generate(3)

# ====================================================================
# PART 2: IN-MEMORY TRANSFER (Replaces file I/O)
# ====================================================================

# Convert GMSH model to DOLFINx mesh and tags IN MEMORY
print("Converting GMSH model to DOLFINx mesh...")
domain, cell_tags, facet_tags = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3)
gmsh.finalize()

# ====================================================================
# PART 3: SOLVER (from solve_eigenvalue.py)
# ====================================================================
print("Setting up eigenvalue problem...")

# Create function space
k = 1
V = fem.functionspace(domain, ("Nedelec 1st kind H(curl)", k))

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define forms
a = fem.form(ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
m = fem.form(ufl.inner(u, v) * ufl.dx)

# Assemble matrices
A = fem.petsc.assemble_matrix(a)
A.setOption(PETSc.Mat.Option.HERMITIAN, True)
A.assemble()

M = fem.petsc.assemble_matrix(m)
M.setOption(PETSc.Mat.Option.HERMITIAN, True)
M.assemble()

# Create boundary condition (using the in-memory facet_tags)
boundary_facets = facet_tags.find(boundary_marker)
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)

# Apply boundary condition
A_bc = fem.petsc.assemble_matrix(a, bcs=[bc])
A_bc.setOption(PETSc.Mat.Option.HERMITIAN, True)
A_bc.assemble()

M_bc = fem.petsc.assemble_matrix(m, bcs=[bc])
M_bc.setOption(PETSc.Mat.Option.HERMITIAN, True)
M_bc.assemble()

# Set up eigensolver
print("Solving eigenvalue problem...")
Eps = SLEPc.EPS().create(MPI.COMM_WORLD)
Eps.setOperators(A_bc, M_bc)
Eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
Eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
Eps.setTarget(10.0)
Eps.setDimensions(nev=6)
Eps.solve()

# Post-process and save solution
nconv = Eps.getConverged()
print(f"Number of converged eigenpairs: {nconv}")
if nconv > 0:
    # Get first eigenpair
    omega_sqr, _ = Eps.getEigenpair(0)
    omega = np.sqrt(omega_sqr) # <-- Bug fix (was omega_s)
    print(f"Computed resonant frequency: {omega}")

    # Get eigenvector
    u_h = fem.Function(V)
    Eps.getEigenvector(0, u_h.vector)
    u_h.x.scatter_forward()

    # Save eigenvector to file
    print("Saving eigenvector to eigenvector.bp...")
    with io.VTXWriter(MPI.COMM_WORLD, "eigenvector.bp", [u_h], "bp4") as f:
        f.write(0.0)

print("Script finished successfully.")
