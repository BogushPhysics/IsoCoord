import numpy as np
from numpy.typing import NDArray
from .surface import Mesh


class Geometry2D:
    mesh: Mesh

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.__perform_calculations()

    def __perform_calculations(self) -> None:
        U = self.mesh.xi_array[:, :, 0]
        V = self.mesh.xi_array[:, :, 1]
        X = self.mesh.xyz_array[:, :, 0]
        Y = self.mesh.xyz_array[:, :, 1]
        Z = self.mesh.xyz_array[:, :, 2]

        dX1 = Dx(X)
        dY1 = Dx(Y)
        dZ1 = Dx(Z)
        dU = Dx(U)

        dXdU = dX1 / dU
        dYdU = dY1 / dU
        dZdU = dZ1 / dU

        self.eU = np.stack((dXdU, dYdU, dZdU), axis=-1)

        dX2 = Dy(X)
        dY2 = Dy(Y)
        dZ2 = Dy(Z)
        dV = Dy(V)

        dXdV = dX2 / dV
        dYdV = dY2 / dV
        dZdV = dZ2 / dV

        self.eV = np.stack((dXdV, dYdV, dZdV), axis=-1)

        self.eU_norm = np.sqrt(np.sum(self.eU * self.eU, axis=2))
        self.eV_norm = np.sqrt(np.sum(self.eV * self.eV, axis=2))

        self.eU_normalized = self.eU / self.eU_norm[:, :, np.newaxis]
        self.eV_normalized = self.eV / self.eV_norm[:, :, np.newaxis]

        self.normal = np.cross(self.eU_normalized, self.eV_normalized)

        self.factor = 0.5 * (self.eU_norm * self.eU_norm +
                             self.eV_norm * self.eV_norm)
        self.error_factor = 0.5 * \
            (self.eU_norm * self.eU_norm - self.eV_norm * self.eV_norm)

        exp_factor = 0.5 * np.log(self.factor)
        f_uu = Dxx(exp_factor) / dU / dU
        f_vv = Dyy(exp_factor) / dV / dV
        self._scalar_curvature_f = - 2 * (f_uu + f_vv) / self.factor

        x_uu = Dxx(X) / dU / dU
        y_uu = Dxx(Y) / dU / dU
        z_uu = Dxx(Z) / dU / dU

        x_uv = Dxy(X) / dV / dU
        y_uv = Dxy(Y) / dV / dU
        z_uv = Dxy(Z) / dV / dU

        x_vv = Dyy(X) / dV / dV
        y_vv = Dyy(Y) / dV / dV
        z_vv = Dyy(Z) / dV / dV

        self.chi_uu = -(x_uu * self.normal[:, :, 0] + y_uu *
                        self.normal[:, :, 1] + z_uu * self.normal[:, :, 2])
        self.chi_uv = -(x_uv * self.normal[:, :, 0] + y_uv *
                        self.normal[:, :, 1] + z_uv * self.normal[:, :, 2])
        self.chi_vv = -(x_vv * self.normal[:, :, 0] + y_vv *
                        self.normal[:, :, 1] + z_vv * self.normal[:, :, 2])

        self.k1 = 0.5 * (self.chi_uu + self.chi_vv + np.sqrt(4.0 * np.power(
            self.chi_uv, 2) + np.power(self.chi_uu - self.chi_vv, 2))) / self.factor
        self.k2 = 0.5 * (self.chi_uu + self.chi_vv - np.sqrt(4.0 * np.power(
            self.chi_uv, 2) + np.power(self.chi_uu - self.chi_vv, 2))) / self.factor
        self.gauss_curvature = self.k1 * self.k2
        self.mean_curvature = 0.5 * (self.k1 + self.k2)
        self.diff_curvature = self.k1 - self.k2


def Dx(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.gradient(array, axis=1, edge_order=2)


def Dy(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.gradient(array, axis=0, edge_order=2)


def Dxx(array: NDArray[np.float64]) -> NDArray[np.float64]:
    result = array[:, 2:] - 2.0 * array[:, 1:-1] + array[:, :-2]
    result_left = 2.0 * array[:, 0] - 5.0 * \
        array[:, 1] + 4.0 * array[:, 2] - array[:, 3]
    result_right = 2.0 * array[:, -1] - 5.0 * \
        array[:, -2] + 4.0 * array[:, -3] - array[:, -4]
    result = np.column_stack([result_left, result, result_right])
    return result


def Dyy(array: NDArray[np.float64]) -> NDArray[np.float64]:
    result = array[2:, :] - 2.0 * array[1:-1, :] + array[:-2, :]
    result_left = 2.0 * array[0, :] - 5.0 * \
        array[1, :] + 4.0 * array[2, :] - array[3, :]
    result_right = 2.0 * array[-1, :] - 5.0 * \
        array[-2, :] + 4.0 * array[-3, :] - array[-4, :]
    result = np.row_stack([result_left, result, result_right])
    return result


def Dxy(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return Dy(Dx(array))
