# maxwell.py — Variable Reference and Modification Guide

This document maps every variable in `maxwell.py` to its purpose, location,
and—where applicable—notes on what to change as the research evolves.

---

## Variable Map

| Variable | Line | Type | Description |
|----------|------|------|-------------|
| `comm` | 14 | MPI communicator | Parallel communication handle. No need to change. |
| `domain` | 17 | dolfinx Mesh | The computational domain (unit cube). |
| `fdim` | 18 | int | Facet dimension (2 for a 3D mesh). Derived from the mesh. |
| `on_x1` | 21 | function | Boundary filter: selects faces where x = 1. |
| `on_boundary` | 22 | function | Boundary filter: selects all boundary faces. |
| `facets_x1` | 24 | numpy array | Array of face IDs on the x = 1 plane (Gamma_d). |
| `facets_all` | 25 | numpy array | Array of all boundary face IDs. |
| `marker` | 29 | numpy array | Integer tag per boundary face. 1 = Gamma_d, 2 = PEC. |
| `facet_tags` | 32 | dolfinx MeshTags | Mesh-attached version of the marker array. |
| `ds` | 33 | ufl Measure | Surface integration measure. `ds(1)` = Gamma_d, `ds(2)` = PEC faces. |
| `curl_el` | 36 | basix element | Nedelec first-kind H(curl) element definition. |
| `V` | 37 | dolfinx FunctionSpace | The H(curl) function space for the electric field. |
| `E` | 38 | ufl TrialFunction | The unknown electric field (what we solve for). |
| `v` | 38 | ufl TestFunction | The test function (used to build the weak form). |
| `omega` | 41 | float | Angular frequency (rad/s). |
| `mu` | 41 | float | Magnetic permeability. |
| `eps` | 41 | float | Electric permittivity. |
| `eta` | 42 | complex (PETSc scalar) | Impedance parameter on Gamma_d. |
| `mu_inv` | 43 | float | Precomputed 1/mu. |
| `x` | 46 | ufl SpatialCoordinate | Symbolic reference to (x, y, z) coordinates in the mesh. |
| `f_expr` | 47-50 | ufl vector expression | Symbolic formula for the source term (Gaussian bump). |
| `V_dg` | 51 | dolfinx FunctionSpace | Vector DG space for representing the source. |
| `f_fun` | 52 | dolfinx Function | The source term with actual numerical values. |
| `n` | 56 | ufl FacetNormal | Outward unit normal vector on boundary faces. |
| `curlE` | 57 | ufl expression | Symbolic curl of the trial function E. |
| `curlv` | 57 | ufl expression | Symbolic curl of the test function v. |
| `tan_inner` | 59 | ufl expression | Inner product of tangential components: (n x E) . conj(n x v). |
| `a` | 61-63 | ufl form | Bilinear form (becomes the matrix A in Ax = b). |
| `L` | 64 | ufl form | Linear form (becomes the right-hand side b in Ax = b). |
| `zero` | 67 | dolfinx Function | Zero function used to enforce PEC (E_tangential = 0). |
| `bc_pec` | 68 | dolfinx DirichletBC | The PEC boundary condition object. |
| `problem` | 71 | dolfinx LinearProblem | The assembled linear system and solver configuration. |
| `E_h` | 74 | dolfinx Function | The computed electric field solution. |
| `P` | 77 | float | Dissipated power through Gamma_d (diagnostic quantity). |
| `V_vis` | 88 | dolfinx FunctionSpace | Vector Lagrange space for ParaView-compatible output. |
| `E_vis` | 91 | dolfinx Function | Real part of E_h, interpolated for visualization. |

---

## Where to Make Changes

### 1. Impedance Parameter eta — Line 42

```python
eta = PETSc.ScalarType(1.0 + 0.0j)
```

**Current state:** Simplified to purely real (no imaginary part).
The original used `eta = 1.0 + 0.2j`.

**What to do:** Replace with the frequency-dependent model from
equation 1.6 in your proposal notes. The requirement is that
Re(eta) > 0 (the real part must stay positive for the problem to be
well-posed and physically meaningful).

**What it affects:**
- Line 63 — appears in the impedance boundary term of the bilinear form `a`
- Line 77 — appears in the dissipation diagnostic `P`

When eta has a nonzero imaginary part, the impedance boundary becomes
partially reactive (reflects some energy back) instead of purely absorbing.

**Example — restoring the original:**
```python
eta = PETSc.ScalarType(1.0 + 0.2j)
```

**Example — making eta depend on omega:**
```python
# Replace line 42 with a model function, e.g.:
eta = PETSc.ScalarType(some_function(omega))
```

---

### 2. Impedance Boundary Location (Gamma_d) — Line 21

```python
def on_x1(x): return np.isclose(x[0], 1.0)
```

**Current state:** Gamma_d is the single face where x = 1.

**What to do:** Change this filter function to select different
faces or combinations of faces. The function receives an array of
coordinates and must return a boolean array.

**What it affects:**
- Line 24 — `facets_x1` is built from this filter
- Lines 29-31 — the marker array assigns tag 1 to these faces
- Line 63 — `ds(1)` integrates the impedance term over these faces
- Line 77 — `ds(1)` integrates the dissipation diagnostic over these faces

**Examples:**
```python
# Impedance on the x=0 face instead:
def on_x1(x): return np.isclose(x[0], 0.0)

# Impedance on both x=0 and x=1:
def on_x1(x): return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

# Impedance on the top face (z=1):
def on_x1(x): return np.isclose(x[2], 1.0)
```

No other code needs to change — the marker/tag system propagates
the choice automatically through `ds(1)`.

---

### 3. Element Degree (Polynomial Order) — Line 36

```python
curl_el = basix.ufl.element("N1curl", domain.topology.cell_name(), 1)
```

**Current state:** Degree 1 (linear). This is the coarsest
approximation.

**What to do:** Increase the final argument to 2, 3, etc. for
higher-order elements and more accuracy per element. This is useful
for convergence studies (verifying the solution improves as you
refine).

**What it affects:**
- Line 37 — the function space `V` will have more degrees of freedom
- Lines 71-74 — the linear system becomes larger and takes longer to solve
- Accuracy of `E_h` improves

**Trade-off:** Degree 2 roughly doubles the DOFs per element.
The direct solver (MUMPS) memory usage grows quickly with problem size.

**Example:**
```python
curl_el = basix.ufl.element("N1curl", domain.topology.cell_name(), 2)
```

---

### 4. Mesh Resolution — Line 17

```python
domain = mesh.create_unit_cube(comm, 16, 16, 16, cell_type=mesh.CellType.tetrahedron)
```

**Current state:** 16 subdivisions along each axis.

**What to do:** Increase for finer meshes (more accuracy) or
decrease for faster testing. The three integers are the number of
subdivisions in x, y, z respectively — they do not need to be equal.

**What it affects:**
- Everything downstream. More elements = more DOFs = larger matrix = slower solve.
- For convergence studies, run with 8, 16, 32, 64 and compare results.

**Rough scaling (tetrahedra with degree-1 Nedelec):**

| Subdivisions | Approx. tetrahedra | Approx. DOFs |
|-------------|-------------------|-------------|
| 8 x 8 x 8 | ~3,000 | ~4,000 |
| 16 x 16 x 16 | ~24,000 | ~30,000 |
| 32 x 32 x 32 | ~196,000 | ~230,000 |

**Example:**
```python
domain = mesh.create_unit_cube(comm, 32, 32, 32, cell_type=mesh.CellType.tetrahedron)
```

---

### 5. Physical Parameters (omega, mu, eps) — Line 41

```python
omega, mu, eps = 10.0, 1.0, 1.0
```

**Current state:** All set to simple constants. mu = eps = 1.0
corresponds to free space (vacuum).

**What to do:**
- **omega**: Change to study different frequencies. Higher omega means
  shorter wavelength, which requires a finer mesh to resolve (rule of
  thumb: ~10 elements per wavelength).
- **mu, eps**: Change to model different materials. Can be made
  spatially varying by replacing with `fem.Function` objects (like
  the source term), though this requires changes to the variational
  form on lines 61-62.

**What it affects:**
- Line 62 — `omega**2 * eps` scales the mass term
- Line 63 — `omega * eta` scales the impedance term
- Line 77 — indirectly, through the solution `E_h`

---

### 6. Source Term — Lines 47-50

```python
f_expr = ufl.as_vector((
    ufl.exp(-200*((x[0]-0.3)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)),
    0.0, 0.0
))
```

**Current state:** A Gaussian bump centered at (0.3, 0.5, 0.5),
pointing in the x-direction.

**What to do:** Modify to change the source's location, width,
direction, or shape entirely. This is the right-hand side f in
the PDE: curl(curl(E)) - omega^2 * eps * E = f.

**Tunable parts:**
- `(0.3, 0.5, 0.5)` — center of the Gaussian (move the source)
- `-200` — controls width (more negative = narrower bump)
- `(gaussian, 0.0, 0.0)` — direction of the source (currently x-only)

**What it affects:**
- Lines 51-53 — interpolation into the DG space
- Line 64 — the linear form `L`

**Examples:**
```python
# Source pointing in z-direction, centered at (0.5, 0.5, 0.5):
f_expr = ufl.as_vector((
    0.0,
    0.0,
    ufl.exp(-200*((x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2))
))

# Wider source (weaker, more spread out):
f_expr = ufl.as_vector((
    ufl.exp(-50*((x[0]-0.3)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)),
    0.0, 0.0
))
```

---

### 7. Solver Configuration — Lines 71-73

```python
petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
```

**Current state:** MUMPS direct solver (exact LU factorization).

**What to do:** For larger meshes (32+ subdivisions), the direct
solver may run out of memory. Switch to an iterative solver at that
point.

**What it affects:** Only the solve step (line 74). The solution
`E_h` should be the same (within solver tolerance) regardless of
method.

**Example — iterative solver for large problems:**
```python
petsc_options={
    "ksp_type": "gmres",
    "pc_type": "asm",         # Additive Schwarz (better than ILU for Maxwell)
    "ksp_rtol": 1e-8,
    "ksp_max_it": 1000,
    "ksp_monitor": None       # prints convergence info each iteration
}
```

---

### 8. Output File — Line 98

```python
with XDMFFile(comm, "E_solution.xdmf", "w") as xdmf:
```

**Current state:** Writes to `E_solution.xdmf` in the working directory.

**What to do:** Change the filename or path as needed. The `.h5`
data file is created automatically alongside the `.xdmf` file.

---

## Data Flow Diagram

```
INPUTS (what you control)                    WHERE IT ENTERS THE EQUATION
====================================         ================================

Mesh resolution (line 17)          ------>   domain
  16 x 16 x 16 tetrahedra

Boundary choice (line 21)          ------>   ds(1) in bilinear form (line 63)
  on_x1: which faces = Gamma_d               ds(1) in diagnostics  (line 77)

Element degree (line 36)           ------>   V (function space, size of system)
  degree = 1

omega (line 41)                    ------>   mass term:      omega^2 * eps  (line 62)
  angular frequency                          impedance term: omega * eta    (line 63)

mu (line 41)                       ------>   curl-curl term: 1/mu           (line 61)
  magnetic permeability

eps (line 41)                      ------>   mass term:      omega^2 * eps  (line 62)
  electric permittivity

eta (line 42)                      ------>   impedance term: 1j*omega*eta   (line 63)
  impedance on Gamma_d                       diagnostics:    Re(eta)        (line 77)

Source f (lines 47-50)             ------>   right-hand side L              (line 64)
  Gaussian bump at (0.3,0.5,0.5)
```

---

## Noted Action Items

| Priority | What | Where | Notes |
|----------|------|-------|-------|
| High | Implement eqn 1.6 for eta(omega) | Line 42 | Replace constant eta with frequency-dependent model from proposal |
| High | Restore imaginary part of eta | Line 42 | Change `0.0j` back to `0.2j` or model value |
| Medium | Experiment with Gamma_d location | Line 21 | Try different boundary faces |
| Medium | Run convergence studies | Lines 17, 36 | Vary mesh resolution and element degree |
| Low | Consider spatially varying mu, eps | Line 41 | Would require restructuring into fem.Function objects |
