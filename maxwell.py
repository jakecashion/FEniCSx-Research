# Maxwell with impedance boundary damping on Gamma_d and PEC elsewhere
# Updated for DOLFINx v0.8/v0.9 compatibility
# Run with: mpirun -n 4 python3 maxwell.py

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags

comm = MPI.COMM_WORLD

# 1) Geometry / mesh
domain = mesh.create_unit_cube(comm, 16, 16, 16, cell_type=mesh.CellType.tetrahedron)
fdim = domain.topology.dim - 1

#defining \Gamma_d as the boundary x=1 (example, can change in the future. Then, PEC BC will be used on the rest of the boundary.)
def on_x1(x): return np.isclose(x[0], 1.0)
def on_boundary(x): return np.full(x.shape[1], True, dtype=bool)

facets_x1 = locate_entities_boundary(domain, fdim, on_x1)
facets_all = locate_entities_boundary(domain, fdim, on_boundary)


#Marking the facets here. 1 corresponds to \Gamma_d, while 2 corresponds to PEC part of the boundary. 
marker = np.zeros(len(facets_all), dtype=np.int32)
marker[np.isin(facets_all, facets_x1)] = 1
marker[marker == 0] = 2
facet_tags = meshtags(domain, fdim, facets_all, marker)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# 2) Function space: H(curl) with Nedelec curl-conforming elements, degree=1 starting point.
curl_el = basix.ufl.element("N1curl", domain.topology.cell_name(), 1)
V = fem.functionspace(domain, curl_el)
E, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# 3) Parameters (complex). Eta: impedance/admittance in scalar model--\eta(\omega) with \Re(\eta)>0, see eqn 1.6 in my proposal notes. Note here \eta = 1+.02 i originally.
omega, mu, eps = 10.0, 1.0, 1.0
eta = PETSc.ScalarType(1.0 + 0.0j)
mu_inv = 1.0 / mu

# 4) Source term (Vector DG space)
x = ufl.SpatialCoordinate(domain)
f_expr = ufl.as_vector((
    ufl.exp(-200*((x[0]-0.3)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)),
    0.0, 0.0
))
V_dg = fem.functionspace(domain, basix.ufl.element("DG", domain.topology.cell_name(), 2, shape=(3,)))
f_fun = fem.Function(V_dg)
f_fun.interpolate(fem.Expression(f_expr, V_dg.element.interpolation_points))

# 5) Variational form
n = ufl.FacetNormal(domain)
curlE, curlv = ufl.curl(E), ufl.curl(v)
# Use ufl.inner for complex conjugation of the test function v
tan_inner = ufl.inner(ufl.cross(n, E), ufl.cross(n, v))

a = (mu_inv * ufl.inner(curlE, curlv) * ufl.dx
     - (omega**2) * eps * ufl.inner(E, v) * ufl.dx
     - 1j * omega * eta * tan_inner * ds(1)) #impedance boundary condition on \Gamma_d
L = ufl.inner(f_fun, v) * ufl.dx

# 6) PEC boundary condition on bdry \setminus \Gamma_d
zero = fem.Function(V)
bc_pec = fem.dirichletbc(zero, fem.locate_dofs_topological(V, fdim, facets_all[marker == 2]))

# 7) Solve linear system (Direct Solver)
problem = LinearProblem(a, L, bcs=[bc_pec], 
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    petsc_options_prefix="maxwell_")
E_h = problem.solve()

# 8) Diagnostics
P = comm.allreduce(fem.assemble_scalar(fem.form(ufl.real(eta) * ufl.inner(ufl.cross(n, E_h), ufl.cross(n, E_h)) * ds(1))), op=MPI.SUM)
if comm.rank == 0: print(f"Dissipation P: {P:.6e}")

# ----------------------------
# 9) Output 
# ----------------------------
try:
    from dolfinx.io import XDMFFile
    
    # 1. Create a standard Vector Lagrange space
    v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
    V_vis = fem.functionspace(domain, v_el)
    
    # 2. Extract ONLY the Real part and convert to a clean Function
    E_vis = fem.Function(V_vis)
    E_vis.name = "Electric_Field"
    
    # This specifically extracts the real part to avoid complex-scalar crashes
    E_vis.interpolate(fem.Expression(ufl.real(E_h), V_vis.element.interpolation_points))

    # 3. Write using the standard XDMF format
    with XDMFFile(comm, "E_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(E_vis)
        
    if comm.rank == 0:
        print("Successfully wrote E_solution.xdmf")

except Exception as e:
    if comm.rank == 0: print(f"Output failed: {e}")