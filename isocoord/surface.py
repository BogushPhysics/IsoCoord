import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, Optional, List, NamedTuple, Self, Union
from scipy.spatial import Delaunay

CallableParametricSurface = \
    Callable[[NDArray[np.float64], NDArray[np.float64]],
             Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]


class Mesh:
    """
    It is considered that `array` is a 3D array representing a 2D array of vectors.
    For example,
    ```
     y_index, x_index = 1, 2
     v = array[y_index, x_index]
     v_y = v[1]
    ```
    where `v` is the radius vector of the following point
    ```
        x x x x
        x x o x
    y   x x x x
    ^   
    | -> x
    ```

    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    size: Tuple[int, int]  # (nx, ny)
    xi_array: NDArray[np.float64]
    xyz_array: NDArray[np.float64]
    triangulation: NDArray[np.int_]
    num_points: int
    num_triangles: int

    def __init__(self, xi_array: NDArray, xyz_array: NDArray, triangulation: NDArray[np.int_]) -> None:
        self.xi_array = xi_array
        self.xyz_array = xyz_array
        self.triangulation = triangulation
        self.size = (self.xi_array.shape[1], self.xi_array.shape[0])
        self.num_points = self.size[0] * self.size[1]
        self.num_triangles = triangulation.shape[0]

    def edges(self, side: str) -> List[int]:
        n_xi, n_eta = self.size
        if side == 'b':
            return list(range(0, n_xi, 1))
        elif side == 'r':
            return list(range(n_xi - 1, n_xi * n_eta, n_xi))
        elif side == 'l':
            return list(range(0, n_xi * (n_eta - 1) + 1, n_xi))
        elif side == 't':
            return list(range(n_xi * (n_eta - 1), n_xi * n_eta, 1))
        if side == 'b_':
            return list(range(1, n_xi - 1, 1))
        elif side == 'r_':
            return list(range(2 * n_xi - 1, n_xi * (n_eta - 1), n_xi))
        elif side == 'l_':
            return list(range(n_xi, n_xi * (n_eta - 2) + 1, n_xi))
        elif side == 't_':
            return list(range(n_xi * (n_eta - 1) + 1, n_xi * n_eta - 1, 1))
        elif side == 'bl':
            return [0]
        elif side == 'br':
            return [n_xi - 1]
        elif side == 'tl':
            return [n_xi * (n_eta - 1)]
        elif side == 'tr':
            return [n_xi * n_eta - 1]
        else:
            raise ValueError

    def refine(self) -> Self:  # TODO
        raise NotImplementedError

    def xyz_flatten(self) -> NDArray[np.float64]:
        return self.xyz_array.reshape(-1, self.xyz_array.shape[-1])

    def xi_flatten(self) -> NDArray[np.float64]:
        return self.xi_array.reshape(-1, self.xi_array.shape[-1])

    @staticmethod
    def unflatten(array: NDArray[np.float64], size: Tuple[int, int], dimensionality: int) -> NDArray:
        return array.reshape((size[1], size[0], dimensionality))


def generate_mesh(surface: CallableParametricSurface,
                  boundaries: Tuple[Tuple[np.float64, np.float64], Tuple[np.float64, np.float64]],
                  size: Tuple[np.int_, np.int_],
                  triangulizer: Optional[Union[Callable, str]] = None) -> Mesh:

    n_xi, n_eta = size

    x_lim, y_lim = boundaries

    xi = np.linspace(x_lim[0], x_lim[1], n_xi)
    eta = np.linspace(y_lim[0], y_lim[1], n_eta)
    xi, eta = np.meshgrid(xi, eta)
    xi_grid = np.stack((xi, eta), axis=2)

    x, y, z = surface(xi, eta)
    xyz_grid = np.stack((x, y, z), axis=2)

    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    triangulation = None
    if triangulizer is None or triangulizer == 'delaunay':
        triangulation = Delaunay(points[:, :2]).simplices
    elif callable(triangulizer):
        triangulation = triangulizer(points)

    if triangulation is None:
        raise ValueError

    return Mesh(xi_grid, xyz_grid, triangulation)
