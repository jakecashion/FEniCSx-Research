# Solving a boundary value problem in FEniCSx consists of the following steps:

#1. Identify the computational domain 
#2. the PDE,
#3. and its corresponding boundary conditions and source terms 

# Reformulate the PDE as a finite element variational problem.

# Write a Python program defining the computational domain, the boundary conditions, the variational problem, and the source terms, using FEniCSx.

# Run the Python program to solve the boundary-value problem. Optionally, you can extend the program to derive quantities such as fluxes and averages,
# and visualize the results.

#Builds The Square Mesh: [0,1] x [0,1]
from mpi4py import MPI
from dolfinx import mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

#Defining the Function Space:
#Step 1: Create the function space, V:
from dolfinx.fem import functionspace
V = functionspace(domain, ("Lagrange", 1))
from dolfinx import fem
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

#Step 2: With the boundary data we now have, apply this to the boundary values that are on the boundary of the discrete domain:

import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

#Step 3: Creating the boundary condition
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

#Defining the Trial and Test Function
import ufl
#FEniCSx uses UFL to define the varitional formulations
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#Defining the Source Term: (f=-6)
from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))

#Defining the variational problem
#variational formulas translate very similar to python code, making it very easy to solve complicated PDE's
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

#Forming and solving the linear system:
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

#Computing the error

#Compares the finite solution: u, with the exact solution.
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

#Two errors to define:

#L^2-Norm Error. FEniCSx uses UFL to define the L^2-error, and uses the dolfin.fem.assemble_scalar function to complete the scalar value
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

#Maximum error at any degree of freedom
error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

#Plotting the Mesh with PyVista:
import pyvista
print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot
# pyvista.start_xvfb() <- Not Needed on a MacOS
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")

#Plotting the function using PyVista:

#Goal: Plotting the function, uh
#Step 1: Creating a mesh based on the dof coordinates for the function space V.
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

#Step 2: creating the grid and adding the dof-values to the mesh.
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()

#Step 3: 3D Plotting:
warped = u_grid.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()

#External Post-Processing:
from dolfinx import io
from pathlib import Path
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)