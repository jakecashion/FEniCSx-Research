import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from dolfinx import fem, io, mesh

# Create mesh
L = 1.0
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, L, L])],
    [16, 16, 16],
    cell_type=mesh.CellType.hexahedron,
)

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

# Create boundary condition
boundary_facets = mesh.locate_entities_boundary(
    domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True)
)
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
Eps = SLEPc.EPS().create(MPI.COMM_WORLD)
Eps.setOperators(A_bc, M_bc)
Eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
Eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
Eps.setTarget(10.0)
Eps.setDimensions(nev=6)
Eps.solve()

# Post-process and save solution
nconv = Eps.getConverged()
if nconv > 0:
    # Get first eigenpair
    omega_sqr, _ = Eps.getEigenpair(0)
    omega = np.sqrt(omega_sqr)
    print(f"Computed resonant frequency: {omega}")

    # Get eigenvector
    u_h = fem.Function(V)
    Eps.getEigenvector(0, u_h.vector)
    u_h.x.scatter_forward()

    # Save eigenvector
    with io.VTXWriter(MPI.COMM_WORLD, "eigenvector.bp", [u_h], "bp4") as f:
        f.write(0.0)