from typing import Dict, Tuple
from numpy.typing import NDArray
import numpy as np
from .surface import CallableParametricSurface, Mesh
from .constraints import ConstraintSystem, ConstraintType
from .lscm import LscmSolver


class IsoCoordinateFinder:
    surface: CallableParametricSurface
    constraints: ConstraintSystem
    xyz_mesh: Mesh
    uv_mesh: Mesh
    target_grid: NDArray
    grid_size: Tuple[int, int]
    n_points: int
    _first_run: bool = True

    def __init__(self, mesh: Mesh, surface: CallableParametricSurface,
                 constraints: ConstraintSystem):
        self.xyz_mesh = Mesh(
            mesh.xi_array.copy(),
            mesh.xyz_array.copy(),
            mesh.triangulation.copy()
        )
        self.uv_mesh = Mesh(
            mesh.xi_array.copy(),
            mesh.xyz_array.copy(),
            mesh.triangulation.copy()
        )
        self.surface = surface
        self.constraints = constraints
        self.grid_size = self.xyz_mesh.size
        self.n_points = self.grid_size[0] * self.grid_size[1]

    def train(self,
              max_iter: int = 100,
              tol: float = 0,
              alpha: float = 1.0,
              method: str = "minres",
              method_kwargs: Dict = {},
              verbose: bool = False):

        solver = LscmSolver(self.constraints, method, method_kwargs)
        if self._first_run:
            self.uv_mesh = solver.solve(self.uv_mesh)
            self._first_run = False
        self.__update_target_grid()
        
        if verbose:
            from tqdm import tqdm
            progress = tqdm(total=max_iter, desc="Progress", unit="it")

        for _ in range(max_iter):
            self.__gradient_descent(alpha)
            init_points = self.uv_mesh.xi_flatten()
            self.uv_mesh = solver.solve(self.uv_mesh, initpoints=init_points)
            self.__update_target_grid()

            error = self.__error()
            if verbose:
                progress.write(f"Displacement error: {error}")
                progress.update(1)

            if error <= tol:
                break

    def __error(self):

        U = self.uv_mesh.xi_array[:, :, 0]
        V = self.uv_mesh.xi_array[:, :, 1]
        U_target = self.target_grid[:, :, 0]
        V_target = self.target_grid[:, :, 1]

        squared_distance = \
            np.power(U - U_target, 2) + np.power(V - V_target, 2)

        return np.sqrt(np.max(squared_distance))

    def __update_target_grid(self) -> None:
        nx, ny = self.grid_size

        U = self.uv_mesh.xi_array[:, :, 0]
        V = self.uv_mesh.xi_array[:, :, 1]

        U_min = 0.5 * (U[0,0] + U[-1,0])
        U_max = 0.5 * (U[0,-1] + U[-1,-1])
        V_min = 0.5 * (V[0,0] + V[0,-1])
        V_max = 0.5 * (V[-1,0] + V[-1,-1])

        U_target = np.linspace(U_min, U_max, nx)
        V_target = np.linspace(V_min, V_max, ny)
        U_target, V_target = np.meshgrid(U_target, V_target)
        self.target_grid = np.stack((U_target, V_target), axis=2)

    def __gradient_descent(self, alpha: float) -> None:

        Xi = self.xyz_mesh.xi_array[:, :, 0]
        Eta = self.xyz_mesh.xi_array[:, :, 1]
        U = self.uv_mesh.xi_array[:, :, 0]
        V = self.uv_mesh.xi_array[:, :, 1]
        U_target = self.target_grid[:, :, 0]
        V_target = self.target_grid[:, :, 1]

        dXi1 = np.gradient(Xi, axis=1, edge_order=2)
        dEta1 = np.gradient(Eta, axis=1, edge_order=2)
        dU = np.gradient(U, axis=1, edge_order=2)

        dXidU = dXi1 / dU
        dEtadU = dEta1 / dU

        dXi2 = np.gradient(Xi, axis=0, edge_order=2)
        dEta2 = np.gradient(Eta, axis=0, edge_order=2)
        dV = np.gradient(V, axis=0, edge_order=2)

        dXidV = dXi2 / dV
        dEtadV = dEta2 / dV

        if self.constraints.constraint_type in {ConstraintType.FIXED_RECTANGLE, ConstraintType.FLEXIBLE_RECTANGLE}:
            dXidV[0, :] = 0
            dXidU[0, :] = 0
            dEtadV[0, :] = 0
            dEtadU[0, :] = 0
            dXidV[-1, :] = 0
            dXidU[-1, :] = 0
            dEtadV[-1, :] = 0
            dEtadU[-1, :] = 0
            dXidV[:, 0] = 0
            dXidU[:, 0] = 0
            dEtadV[:, 0] = 0
            dEtadU[:, 0] = 0
            dXidV[:, -1] = 0
            dXidU[:, -1] = 0
            dEtadV[:, -1] = 0
            dEtadU[:, -1] = 0
        else:
            dXidV[0, 0] = 0
            dXidU[0, 0] = 0
            dEtadV[0, 0] = 0
            dEtadU[0, 0] = 0
            dXidV[-1, 0] = 0
            dXidU[-1, 0] = 0
            dEtadV[-1, 0] = 0
            dEtadU[-1, 0] = 0
            dXidV[0, -1] = 0
            dXidU[0, -1] = 0
            dEtadV[0, -1] = 0
            dEtadU[0, -1] = 0
            dXidV[-1, -1] = 0
            dXidU[-1, -1] = 0
            dEtadV[-1, -1] = 0
            dEtadU[-1, -1] = 0

        Xi_new = Xi + alpha * ((U_target - U) * dXidU + (V_target - V) * dXidV)
        Eta_new = Eta + alpha * \
            ((U_target - U) * dEtadU + (V_target - V) * dEtadV)

        X_new, Y_new, Z_new = self.surface(Xi_new, Eta_new)

        XiEta_new = np.stack((Xi_new, Eta_new), axis=2)
        XYZ_new = np.stack((X_new, Y_new, Z_new), axis=2)

        self.xyz_mesh.xi_array = XiEta_new
        self.xyz_mesh.xyz_array = XYZ_new
        self.uv_mesh.xyz_array = XYZ_new
