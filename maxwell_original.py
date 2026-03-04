# Maxwell with impedance boundary damping on Gamma_d and PEC elsewhere
# Requires: dolfinx, mpi4py, petsc4py
# Run with: python3 this_file.py

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags


comm = MPI.COMM_WORLD

# ----------------------------
# 1) Geometry / mesh
# ----------------------------
# Unit cube mesh (replace with your own mesh as needed)
domain = mesh.create_unit_cube(comm, 16, 16, 16, cell_type=mesh.CellType.tetrahedron)

tdim = domain.topology.dim
fdim = tdim - 1

# Define Gamma_d as the boundary x=1 (example).
# PEC on the rest of boundary.
def on_x1(x):
    return np.isclose(x[0], 1.0)

def on_boundary(x):
    # locate_entities_boundary already restricts to boundary
    return np.full(x.shape[1], True, dtype=bool)

facets_x1 = locate_entities_boundary(domain, fdim, on_x1)
facets_all = locate_entities_boundary(domain, fdim, on_boundary)

# Mark facets: 1 for Gamma_d, 2 for PEC part
marker = np.zeros(len(facets_all), dtype=np.int32)
marker[np.isin(facets_all, facets_x1)] = 1
marker[marker == 0] = 2

facet_tags = meshtags(domain, fdim, facets_all, marker)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# ----------------------------
# 2) Function space: H(curl)
# ----------------------------
# Nedelec (first kind) curl-conforming elements
# degree=1 is a common starting point
V = fem.functionspace(domain, ("N1curl", 1))

E = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# ----------------------------
# 3) Parameters (complex)
# ----------------------------
omega = 10.0        # rad/s
mu    = 1.0         # can be spatially varying, but keep scalar here
eps   = 1.0         # same

# Impedance/admittance scalar model: eta(omega) with Re(eta) > 0
# Example: eta = eta0 + i*eta1
eta0 = 1.0
eta1 = 0.2
eta  = PETSc.ScalarType(eta0 + 1j * eta1)

mu_inv = 1.0 / mu

# ----------------------------
# 4) Source term
# ----------------------------
# Example forcing: a localized current-like RHS in the volume
# Replace with your model (or set to zero).
x = ufl.SpatialCoordinate(domain)
f_expr = ufl.as_vector((
    ufl.exp(-200*((x[0]-0.3)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)),
    0*x[1],
    0*x[2]
))
f = fem.Expression(f_expr, fem.functionspace(domain, ("DG", 2)).element.interpolation_points())
f_fun = fem.Function(f.function_space)
f_fun.interpolate(f)

# ----------------------------
# 5) Variational form
# ----------------------------
n = ufl.FacetNormal(domain)
curlE = ufl.curl(E)
curlv = ufl.curl(v)

# Tangential trace inner product: (n×E)·(n×v)
tan_inner = ufl.dot(ufl.cross(n, E), ufl.cross(n, v))

a = (mu_inv * ufl.inner(curlE, curlv) * ufl.dx
     - (omega**2) * eps * ufl.inner(E, v) * ufl.dx
     - 1j * omega * eta * tan_inner * ds(1))   # impedance on Gamma_d

L = ufl.inner(f_fun, v) * ufl.dx

# ----------------------------
# 6) PEC boundary condition on (boundary \ Gamma_d)
# ----------------------------
# For Nedelec spaces, setting dofs on PEC facets enforces n×E=0.
zero = fem.Function(V)
zero.x.array[:] = 0.0

pec_facets = facets_all[marker == 2]
pec_dofs   = fem.locate_dofs_topological(V, fdim, pec_facets)
bc_pec     = fem.dirichletbc(zero, pec_dofs)

# ----------------------------
# 7) Solve linear system
# ----------------------------
problem = LinearProblem(
    a, L, bcs=[bc_pec],
    petsc_options={
        "ksp_type":   "gmres",
        "pc_type":    "ilu",
        "ksp_rtol":   1e-8,
        "ksp_max_it": 500
    },
)

E_h = problem.solve()
E_h.name = "E"

# ----------------------------
# 8) (Optional) Diagnostics: boundary dissipation on Gamma_d
# ----------------------------
# A common measure is proportional to Re(eta) * ||n×E||^2 on Gamma_d.
# Compute: P = ∫_{Gamma_d} Re(eta) |n×E|^2 ds
Et2    = ufl.inner(ufl.cross(n, E_h), ufl.cross(n, E_h))
P_form = ufl.real(eta) * Et2 * ds(1)

P = fem.assemble_scalar(fem.form(P_form))
P = comm.allreduce(P, op=MPI.SUM)

if comm.rank == 0:
    print(f"Computed boundary dissipation proxy P = ∫ Re(eta)|n×E|^2 ds on Gamma_d:  {P:.6e}")

# ----------------------------
# 9) Output (optional)
# ----------------------------
# Write to XDMF for visualization (ParaView)
try:
    from dolfinx.io import XDMFFile
    with XDMFFile(comm, "E_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(E_h)
    if comm.rank == 0:
        print("Wrote E_solution.xdmf")
except Exception as e:
    if comm.rank == 0:
        print("XDMF output skipped:", e)
