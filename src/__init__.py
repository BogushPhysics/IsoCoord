# encoding: utf-8
"""
**IsoCoord**

A Python package for finding isothermal coordinates.

"""


__author__ = 'Igor Bogush'
__email__ = 'bogush94@gmail.com'
__version__ = '0.1.0'
__url__ = "https://github.com/BogushPhysics/IsoCoord/tree/main"
__description__ = "A Python package for finding isothermal coordinates."
__license__ = "MIT"


from .surface import CallableParametricSurface, Mesh, generate_mesh
from .constraints import ConstraintElement, Constraint, ConstraintSystem, \
    generate_2point_constraint, generate_fixed_corner_constraint, \
    generate_flexible_corner_constraint, generate_fixed_rectangle_constraint, \
    generate_flexible_rectangle_constraint
from .isothermal import IsoCoordinateFinder
from .lscm import LscmSolver
from .geometry import Geometry2D


__all__ = [
    "CallableParametricSurface",
    "Mesh",
    "generate_mesh",
    "ConstraintElement",
    "Constraint",
    "ConstraintSystem",
    "IsoCoordinateFinder",
    "LscmSolver",
    "Geometry2D",
    "generate_2point_constraint",
    "generate_fixed_corner_constraint",
    "generate_flexible_corner_constraint",
    "generate_fixed_rectangle_constraint",
    "generate_flexible_rectangle_constraint"
]
