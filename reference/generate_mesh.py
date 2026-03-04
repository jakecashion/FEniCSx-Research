import gmsh
import json
import numpy as np
import sys

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

# Create box
box = model.occ.addBox(c_x - L / 2, c_y - L / 2, c_z - L / 2, L, L, L)
# Create cylinders
cyl_ext = model.occ.addCylinder(c_x, c_y, c_z - l_ext / 2, 0, 0, l_ext, r_ext)
cyl_int = model.occ.addCylinder(c_x, c_y, c_z - l_int / 2, 0, 0, l_int, r_int)

# Cut internal cylinder from external cylinder
cut = model.occ.cut([(3, cyl_ext)], [(3, cyl_int)], removeTool=True)

# Fuse box and external cylinder
fusion = model.occ.fuse([(3, box)], cut[0])
model.occ.synchronize()

# Add physical groups
volumes = model.getEntities(3)
assert len(volumes) == 1
model.addPhysicalGroup(3, [volumes[0][1]], 1)
model.setPhysicalName(3, 1, "cavity")

surfaces = model.occ.getEntities(2)
boundary_marker = 2
boundary_surfaces = []
for surface in surfaces:
    com = model.occ.getCenterOfMass(surface[0], surface[1])
    if not np.allclose(com, [0, 0, 0]):
        boundary_surfaces.append(surface[1])
model.addPhysicalGroup(2, boundary_surfaces, boundary_marker)
model.setPhysicalName(2, boundary_marker, "boundary")

# Set mesh size
lc = 0.1
model.mesh.setSize(model.getEntities(0), lc)

# Generate mesh
model.mesh.generate(3)

# Save mesh
gmsh.write("mesh.msh")

# Save mesh tags
tdim = 3
cell_entities = model.getEntities(tdim)
cell_tags = {}
for cell in cell_entities:
    cell_tags[int(cell[1])] = int(model.getPhysicalGroupsForEntity(tdim, cell[1])[0])

with open("cell_tags.json", "w") as f:
    json.dump(cell_tags, f)

fdim = 2
face_entities = model.getEntities(fdim)
face_tags = {}
for face in face_entities:
    face_tags[int(face[1])] = int(model.getPhysicalGroupsForEntity(fdim, face[1])[0])

with open("mesh_tags.json", "w") as f:
    json.dump(face_tags, f)

gmsh.finalize()
