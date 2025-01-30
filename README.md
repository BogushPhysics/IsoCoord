# IsoCoord

A Python library for computing isothermal coordinates using conformal mappings


## Motivation

Many numerical algorithms in physics rely on finite difference methods (FDM), particularly for computing physical properties in thin rectangular planar membranes. While FDM is well-suited for rectangular domains, it becomes less suited for membranes of arbitrary 3D geometries. Moreover, FDM can be highly efficient in GPU-based simulations.

__IsoCoord__ facilitates physical simulations of membranes with complex 3D geometries by transforming the problem into a flat domain using conformal mappings and isothermal coordinates. It also computes essential geometric quantities, such as the conformal factor and curvatures. In many cases, equations for 3D membranes can be reformulated in the form of flat ones by incorporating these geometric quantities in a simple multiplicative manner.


## Application

The mathematical details and examples of the application to Schrodinger and time-dependent Ginzburg-Landau equations are given in [arXiv:2412.15741](https://arxiv.org/abs/2412.15741).


## Install `IsoCoord`

```bash
git clone https://github.com/BogushPhysics/IsoCoord.git
cd IsoCoord
pip install -e "."
```

## Examples

```python
import isocoord as ic
import numpy as np

# define the surface
def surface(x, y):
    r2 = x * x + y * y
    r = np.sqrt(r2)
    return x, y, np.exp(-4.0*(r - 2.0)**2) * (1.0 - np.exp(-4.0 * (np.arctan2(y, x))**2))

# define the grid size
nx, ny = 201, 201

# create the mesh and constraints
mesh = ic.generate_mesh(surface, ((-5, 5), (-5, 5)), (nx, ny))
cs = ic.generate_fixed_rectangle_constraint(mesh, 1)

# create the model to be trained 
isofinder = ic.IsoCoordinateFinder(mesh, surface, cs)

# train the model
isofinder.train(10, tol=1e-4, alpha=0.2, method="minres", method_kwargs=dict({'tol': 1e-7}), verbose=False)

# get isothermal coordinates and the corresponding 3D coordinates
uv_coordinates, xyz_coordinates = mesh.xi_flatten(), mesh.xyz_flatten()
```
