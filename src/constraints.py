from typing import NamedTuple, List
import numpy as np
from .surface import Mesh

# each ConstraintElement is (point, coordiante, coefficient)
# each constraint is List[ConstraintElement], rhs


class ConstraintElement(NamedTuple):
    point: int
    idx: int  # coordinate index: 0 for x, 1 for y
    coef: float


class Constraint(NamedTuple):
    """_summary_

    coef_1 * p_{c_1} + ... + coef_n * p_{c_n} == rhs
    """
    elements: List[ConstraintElement]
    rhs: float

    @staticmethod
    def _singleElement(point, idx, value) -> "Constraint":
        return Constraint(
            elements=[ConstraintElement(point, idx, 1.0)],
            rhs=value
        )

    @staticmethod
    def _diffOneElement(point1, point2, idx) -> "Constraint":
        return Constraint(
            elements=[
                ConstraintElement(point1, idx, 1.0),
                ConstraintElement(point2, idx, -1.0)
            ],
            rhs=0.0
        )

    @staticmethod
    def _diffTwoElement(point1, point2, point3, idx) -> "Constraint":
        return Constraint(
            elements=[
                ConstraintElement(point1, idx, 1.0),
                ConstraintElement(point2, idx, -2.0),
                ConstraintElement(point3, idx, 1.0)
            ],
            rhs=0.0
        )


class ConstraintSystem:
    constraints: List[Constraint]
    total_points: int
    total_triangles: int
    total_constraints: int

    def __init__(self, constraints: List[Constraint], total_points: int, total_triangles: int):
        self.constraints = constraints
        self.total_points = total_points
        self.total_triangles = total_triangles
        self.total_constraints = len(constraints)


def generate_2point_constraint(mesh: Mesh, points: List[int],
                               coordinates: List[List[float]]) -> ConstraintSystem:
    _constraints = [
        Constraint._singleElement(points[0], 0, coordinates[0][0]),
        Constraint._singleElement(points[0], 1, coordinates[0][1]),
        Constraint._singleElement(points[1], 0, coordinates[1][0]),
        Constraint._singleElement(points[1], 1, coordinates[1][1])
    ]
    return ConstraintSystem(_constraints, mesh.num_points, mesh.num_triangles)


def generate_fixed_corner_constraint(mesh: Mesh, ratio: float) -> ConstraintSystem:
    bl = mesh.edges('bl')[0]
    br = mesh.edges('br')[0]
    tr = mesh.edges('tr')[0]
    tl = mesh.edges('tl')[0]

    _constraints = [
        Constraint._singleElement(bl, 0, 0.0),
        Constraint._singleElement(bl, 1, 0.0),
        Constraint._singleElement(br, 0, 1.0),
        Constraint._singleElement(br, 1, 0.0),
        Constraint._singleElement(tr, 0, 1.0),
        Constraint._singleElement(tr, 1, ratio),
        Constraint._singleElement(tl, 0, 0),
        Constraint._singleElement(tl, 1, ratio)
    ]
    return ConstraintSystem(_constraints, mesh.num_points, mesh.num_triangles)


def generate_flexible_corner_constraint(mesh: Mesh) -> ConstraintSystem:
    bl = mesh.edges('bl')[0]
    br = mesh.edges('br')[0]
    tr = mesh.edges('tr')[0]
    tl = mesh.edges('tl')[0]

    _constraints = [
        Constraint._singleElement(bl, 0, 0.0),
        Constraint._singleElement(bl, 1, 0.0),
        Constraint._singleElement(br, 0, 1.0),
        Constraint._singleElement(br, 1, 0.0),
        Constraint._singleElement(tl, 0, 0.0),
        Constraint._singleElement(tr, 0, 1.0),
        Constraint._diffOneElement(tr, tl, 1)
    ]
    return ConstraintSystem(_constraints, mesh.num_points, mesh.num_triangles)


def generate_fixed_rectangle_constraint(mesh: Mesh, ratio: float) -> ConstraintSystem:
    edge_b = mesh.edges('b')
    edge_r = mesh.edges('r')
    edge_t = mesh.edges('t')
    edge_l = mesh.edges('l')
    nx, ny = mesh.size

    con_b_0 = [Constraint._singleElement(e, 0, x)
               for e, x in zip(edge_b, np.linspace(0, 1, nx))]
    con_b_1 = [Constraint._singleElement(e, 1, 0.0) for e in edge_b]

    con_t_0 = [Constraint._singleElement(e, 0, x)
               for e, x in zip(edge_t, np.linspace(0, 1, nx))]
    con_t_1 = [Constraint._singleElement(e, 1, ratio) for e in edge_t]

    con_l_0 = [Constraint._singleElement(e, 0, 0.0) for e in edge_l]
    con_l_1 = [Constraint._singleElement(e, 1, y)
               for e, y in zip(edge_l, np.linspace(0, ratio, ny))]

    con_r_0 = [Constraint._singleElement(e, 0, 1.0) for e in edge_r]
    con_r_1 = [Constraint._singleElement(e, 1, y)
               for e, y in zip(edge_r, np.linspace(0, ratio, ny))]

    _constraints = con_b_0 + con_b_1 + con_t_0 + \
        con_t_1 + con_l_0 + con_l_1 + con_r_0 + con_r_1
    return ConstraintSystem(_constraints, mesh.num_points, mesh.num_triangles)


def generate_flexible_rectangle_constraint(mesh: Mesh) -> ConstraintSystem:
    edge_b = mesh.edges('b')
    edge_r = mesh.edges('r')
    edge_t = mesh.edges('t')
    edge_l = mesh.edges('l')
    nx, ny = mesh.size

    con_b_0 = [Constraint._singleElement(e, 0, x)
               for e, x in zip(edge_b, np.linspace(0, 1, nx))]
    con_b_1 = [Constraint._singleElement(e, 1, 0.0) for e in edge_b]

    con_t_0 = [Constraint._singleElement(e, 0, x)
               for e, x in zip(edge_t, np.linspace(0, 1, nx))]
    con_t_1 = [Constraint._diffOneElement(e2, e1, 1)
               for e1, e2 in zip(edge_t[:-1], edge_t[1:])]

    con_l_0 = [Constraint._singleElement(e, 0, 0.0) for e in edge_l]
    con_l_1 = [Constraint._diffTwoElement(e1, e2, e3, 1)
               for e1, e2, e3 in zip(edge_l[:-2], edge_l[1:-1], edge_l[2:])]

    con_r_0 = [Constraint._singleElement(e, 0, 1.0) for e in edge_r]
    con_r_1 = [Constraint._diffTwoElement(e1, e2, e3, 1)
               for e1, e2, e3 in zip(edge_r[:-2], edge_r[1:-1], edge_r[2:])]

    _constraints = con_b_0 + con_b_1 + con_t_0 + \
        con_t_1 + con_l_0 + con_l_1 + con_r_0 + con_r_1
    return ConstraintSystem(_constraints, mesh.num_points, mesh.num_triangles)
