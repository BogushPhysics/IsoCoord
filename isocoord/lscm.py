from typing import Optional, List, Dict, Union, Tuple, Any
from numpy.typing import NDArray
import numpy as np
import scipy as sp
from .surface import Mesh
from .constraints import ConstraintSystem


class LscmSolver:
    lib: str
    method: Optional[str]
    method_kwargs: Optional[Dict]
    constraint_matrix: NDArray
    rhs_vector: NDArray

    def __init__(self,
                 constraints: ConstraintSystem,
                 method: Optional[str] = "minres",
                 method_kwargs: Optional[Dict] = None):
        self.method = method
        self.method_kwargs = method_kwargs
        self.constraint_matrix, self.rhs_vector = \
            self.__generate_constraint_matrix(constraints)

    def solve(self, mesh: Mesh, initpoints: Optional[NDArray[np.float_]] = None) -> Mesh:
        points = mesh.xyz_flatten()
        triangulation = mesh.triangulation

        M = self._prepare_quadratic_sqrt_form(points, triangulation)
        A_complex = M.conjugate().transpose() @ M

        n_points = len(points)

        reA = np.real(A_complex)
        imA = np.imag(A_complex)
        A_real = _trim_small_values(sp.sparse.bmat([
            [reA, -imA],
            [imA, reA]
        ]))

        S_real = self.constraint_matrix
        b_real = self.rhs_vector
        A_full = sp.sparse.bmat(
            [[A_real, S_real.transpose()], [S_real, None]], format='csr')

        if initpoints is None:
            initvals = None
        else:
            initvals = np.concatenate(
                (initpoints[:, 0], initpoints[:, 1], np.zeros(S_real.shape[0]))
            )

        if self.method == 'minres':
            solver = sp.sparse.linalg.minres
        elif self.method == 'gmres':
            solver = sp.sparse.linalg.gmres
        elif self.method == 'qmr':
            solver = sp.sparse.linalg.qmr
        elif self.method == 'bicgstab':
            solver = sp.sparse.linalg.bicgstab
        elif self.method == 'spsolve':
            solution = sp.sparse.linalg.spsolve(A_full, b_real)
        elif self.method == 'factorized':
            _solver = sp.sparse.linalg.factorized(sp.sparse.csc_matrix(A_full))
            solution = _solver(b_real)
        else:
            raise ValueError

        # Solve the quadratic programming problem
        if solver != None:
            _s = solver(A_full, b_real, x0=initvals, **self.method_kwargs)
            solution = _s[0]
        # lagrange = solution[A_real.shape[0]:]
        solution = solution[:A_real.shape[0]]

        u = np.array(solution[:n_points])
        v = np.array(solution[n_points:])
        uv_points = np.column_stack((u, v))

        # solution_transposed = solution.transpose().conjugate()
        # error = np.abs(solution_transposed @ A_real @
        #               solution / (solution_transposed @ solution))

        uv_points = Mesh.unflatten(uv_points, mesh.size, 2)
        new_mesh = Mesh(uv_points, mesh.xyz_array, mesh.triangulation)

        return new_mesh

    def __generate_constraint_matrix(self, cons: ConstraintSystem) -> Tuple[NDArray, NDArray]:
        n = cons.total_points
        S_data: List[float] = []
        S_rows: List[float] = []
        S_cols: List[float] = []
        S_shape = (cons.total_constraints, 2 * n)
        b_data: List[float] = []

        for row, c in enumerate(cons.constraints):
            for e in c.elements:
                S_data.append(e.coef)
                S_rows.append(row)
                S_cols.append(e.point + n * e.idx)
            b_data.append(c.rhs)

        constraint_matrix = sp.sparse.coo_matrix(
            (S_data, (S_rows, S_cols)), shape=S_shape)
        # skip first block in rhs
        rhs_vector = np.array([0] * (2*n) + b_data)

        return constraint_matrix, rhs_vector

    def _triangle_to_complex_vector(self, points: NDArray[np.float_], triangulation: NDArray[np.int_]) -> Tuple[NDArray[np.complex_], NDArray[np.complex_]]:
        pp = points[triangulation.flatten()]

        p0 = pp[::3]
        p1 = pp[1::3]
        p2 = pp[2::3]

        r1 = p1 - p0
        r2 = p2 - p0

        r1_norm = np.sqrt(np.sum(r1 * r1, axis=1))
        e1 = r1 / r1_norm[:, np.newaxis]
        r2p = r2 - e1 * np.sum(r2 * e1, axis=1)[:, np.newaxis]
        r2p_norm = np.sqrt(np.sum(r2p * r2p, axis=1))
        e2 = r2p / r2p_norm[:, np.newaxis]

        z1 = r1_norm + 0j
        z2 = np.sum(r2 * e1, axis=1) + 1j * np.sum(r2 * e2, axis=1)

        return z1, z2

    def _prepare_quadratic_sqrt_form(self, points: NDArray[np.float_], triangulation: NDArray[np.int_]) -> sp.sparse.csr_matrix:

        n_points = points.shape[0]
        n_triangles = triangulation.shape[0]

        z1, z2 = self._triangle_to_complex_vector(points, triangulation)
        w0, w1, w2 = z2 - z1, - z2, z1
        areaSqrt = np.sqrt(0.5 * np.abs(np.imag(z1 * z2.conjugate())))
        w0, w1, w2 = w0 / areaSqrt, w1 / areaSqrt, w2 / areaSqrt

        data = np.column_stack((w0, w1, w2)).flatten()
        rows = np.repeat(range(len(triangulation)), 3)
        cols = triangulation.flatten()

        M = sp.sparse.coo_matrix((data, (rows, cols)),
                                 shape=(n_triangles, n_points)).tocsr()
        return M


def _trim_small_values(sparse_matrix, threshold=1e-15):
    """
    Trim values in a sparse matrix that are below a given threshold.

    Parameters:
    sparse_matrix (scipy.sparse.spmatrix): The input sparse matrix.
    threshold (float): The threshold below which values will be trimmed.

    Returns:
    scipy.sparse.spmatrix: The sparse matrix with small values removed.
    """
    # Convert to COO format for easy manipulation
    coo = sparse_matrix.tocoo()

    # Apply the threshold
    mask = np.abs(coo.data) > threshold

    # Create a new COO matrix with the filtered data
    filtered_coo = sp.sparse.coo_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=sparse_matrix.shape)

    # Convert back to CSR or CSC format
    return filtered_coo.tocsr()
