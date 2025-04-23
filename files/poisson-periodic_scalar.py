
"""

https://github.com/jorgensd/dolfinx_mpc/blob/main/python/demos/demo_periodic_geometrical.py

but adapted to a scalar problem
"""


from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem as fem
import pyvista
from dolfinx import plot
import numpy as np
from dolfinx import default_scalar_type
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dx,
    exp,
    grad,
    inner,
    pi,
    sin,
)
from dolfinx_mpc import LinearProblem, MultiPointConstraint


# Create mesh and finite element
NX = 50
NY = 100
mesh = create_unit_square(MPI.COMM_WORLD, NX, NY)
V = fem.functionspace(mesh, ("Lagrange", 1))
tol = 250 * np.finfo(default_scalar_type).resolution


def dirichletboundary(x):
    return np.logical_or(np.isclose(x[1], 0, atol=tol), 
                         np.isclose(x[1], 1, atol=tol))


# Create Dirichlet boundary condition
facets = locate_entities_boundary(mesh, 1, dirichletboundary)
topological_dofs = fem.locate_dofs_topological(V, 1, facets)
bc = fem.dirichletbc(0., topological_dofs, V)
bcs = [bc]


def periodic_boundary(x):
    return np.isclose(x[0], 1, atol=tol)


def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


mpc = MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcs)
mpc.finalize()

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx

x = SpatialCoordinate(mesh)
dx_ = x[0] - 0.9
dy_ = x[1] - 0.5
# f = as_vector((x[0] * sin(5.0 * pi * x[1]) + 1.0 * exp(-(dx_ * dx_ + dy_ * dy_) / 0.02), 0.3 * x[1]))
f = x[0] * sin(5.0 * pi * x[1]) + exp(-(dx_ * dx_ + dy_ * dy_) / 0.02)
# f = x[0]**4 + 3*x[0] + x[1]**4

rhs = inner(f, v) * dx

# solve
petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
problem = LinearProblem(a, rhs, mpc, bcs=bcs, petsc_options=petsc_options)
uh = problem.solve()

# plots
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
warped = grid.warp_by_scalar(factor=20)
plotter.add_mesh(warped)
plotter.add_mesh(grid, show_edges=True)
plotter.show_axes()

plotter.camera_position = [(0.5, 0.5, 2.5), # position of camera      
                           (0.5, 0.5, 0),   # point which cam looks at
                           (0  , 1  , 0),]  # up-axis

plotter.camera_position = [(2.0, 2.0, 1.5), # position of camera      
                           (0.5, 0.5, 0),   # point which cam looks at
                           (0  , 0  , 1),]  # up-axis
plotter.save_graphic("poisson_periodic_scalar.pdf")
plotter.show()
