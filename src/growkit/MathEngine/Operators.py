"""
Operators Module (isotropy-focused, API-compatible)

Kept function names:
- _gradient_neumann(field, dx, axis)
- gradient(field, dx) -> (gx, gy, gz)
- divergence(ux, uy, uz, dx) -> div
- laplacian(field, dx) -> lap
- laplacian_neumann(p_flat, shape, dx) -> lap_flat
- curl(ux, uy, uz, dx) -> (cx, cy, cz)
- vector_laplacian(ux, uy, uz, dx) -> (lx, ly, lz)
- isotropic_gradient(field, dx) -> |∇φ|
- isotropic_gradient_components(field, dx) -> (gix, giy, giz)

New (optional, non-breaking):
- isotropic_laplacian(field, dx) -> ∇·(isotropic ∇φ)

All operators use Neumann (zero-normal-derivative) boundary behavior.
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = [
    "_gradient_neumann",
    "gradient",
    "divergence",
    "laplacian",
    "laplacian_neumann",
    "curl",
    "vector_laplacian",
    "isotropic_gradient",
    "isotropic_gradient_components",
    "isotropic_laplacian",
]

# ----------------------------
# Axis-aligned core primitives
# ----------------------------

@njit(cache=True, fastmath=True)
def _gradient_neumann(field: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """
    Axis-aligned central-difference gradient with Neumann (zero-flux) BCs.
    2nd-order centered interior; zero normal derivative at domain faces.
    """
    grad = np.zeros_like(field)
    if axis == 0:
        grad[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2.0 * dx)
        grad[0, :, :] = 0.0
        grad[-1, :, :] = 0.0
    elif axis == 1:
        grad[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dx)
        grad[:, 0, :] = 0.0
        grad[:, -1, :] = 0.0
    else:
        grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dx)
        grad[:, :, 0] = 0.0
        grad[:, :, -1] = 0.0
    return grad


def gradient(field: np.ndarray, dx: float):
    """Axis-aligned ∇φ with Neumann BCs (kept for API compatibility)."""
    gx = _gradient_neumann(field, dx, 0)
    gy = _gradient_neumann(field, dx, 1)
    gz = _gradient_neumann(field, dx, 2)
    return gx, gy, gz


def divergence(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float) -> np.ndarray:
    """Axis-aligned ∇·u with Neumann BCs on each component."""
    return (
        _gradient_neumann(ux, dx, 0)
        + _gradient_neumann(uy, dx, 1)
        + _gradient_neumann(uz, dx, 2)
    )


@njit(cache=True, fastmath=True)
def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Axis-aligned ∇²φ with Neumann BCs (classic 6-point stencil, 2nd-order).
    Kept for compatibility/benchmarks.
    """
    lap = np.zeros_like(field)
    inv_dx2 = 1.0 / (dx * dx)

    # x
    lap[1:-1, :, :] += (field[2:, :, :] - 2.0 * field[1:-1, :, :] + field[:-2, :, :]) * inv_dx2
    lap[0, :, :]     += (field[1, :, :] - field[0, :, :]) * inv_dx2
    lap[-1, :, :]    += (field[-2, :, :] - field[-1, :, :]) * inv_dx2

    # y
    lap[:, 1:-1, :] += (field[:, 2:, :] - 2.0 * field[:, 1:-1, :] + field[:, :-2, :]) * inv_dx2
    lap[:, 0, :]    += (field[:, 1, :] - field[:, 0, :]) * inv_dx2
    lap[:, -1, :]   += (field[:, -2, :] - field[:, -1, :]) * inv_dx2

    # z
    lap[:, :, 1:-1] += (field[:, :, 2:] - 2.0 * field[:, :, 1:-1] + field[:, :, :-2]) * inv_dx2
    lap[:, :, 0]    += (field[:, :, 1] - field[:, :, 0]) * inv_dx2
    lap[:, :, -1]   += (field[:, :, -2] - field[:, :, -1]) * inv_dx2

    return lap


def laplacian_neumann(p_flat: np.ndarray, shape: tuple[int, int, int], dx: float) -> np.ndarray:
    """
    Axis-aligned ∇²φ with Neumann BCs for flattened arrays (pressure solver path).
    Returns float64 flattened array to match solver expectations.
    """
    p = p_flat.reshape(shape)
    lap_p = laplacian(p, dx).astype(np.float64, copy=False)
    return lap_p.ravel()


@njit(cache=True, fastmath=True)
def curl(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float):
    """Axis-aligned curl with Neumann BCs on partials."""
    cx = _gradient_neumann(uz, dx, 1) - _gradient_neumann(uy, dx, 2)
    cy = _gradient_neumann(ux, dx, 2) - _gradient_neumann(uz, dx, 0)
    cz = _gradient_neumann(uy, dx, 0) - _gradient_neumann(ux, dx, 1)
    return cx, cy, cz


def vector_laplacian(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float):
    """Axis-aligned vector Laplacian, componentwise."""
    return laplacian(ux, dx), laplacian(uy, dx), laplacian(uz, dx)

# ---------------------------------------
# Isotropic operators (recommended)
# ---------------------------------------

@njit(cache=True, fastmath=True)
def _zero_face_gradients(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray):
    """Enforce Neumann BCs (zero normal derivative) for gradient components at faces."""
    gx[0, :, :] = 0.0
    gx[-1, :, :] = 0.0
    gy[:, 0, :] = 0.0
    gy[:, -1, :] = 0.0
    gz[:, :, 0] = 0.0
    gz[:, :, -1] = 0.0


@njit(cache=True, fastmath=True)
def isotropic_gradient_components(field: np.ndarray, dx: float):
    """
    Isotropic gradient components using a 3×3×3 blended stencil:
      - centered axis diffs + small diagonal blends (xy, xz, yz planes).
    Using the same stencil for ∇ and its divergence reduces grid-aligned artifacts.
    """
    nx, ny, nz = field.shape
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
    gz = np.zeros_like(field)

    inv_2dx = 1.0 / (2.0 * dx)
    inv_2sqrt2dx = 1.0 / (2.0 * np.sqrt(2.0) * dx)
    w_axis = 1.0
    w_diag = 0.10  # diagonal blend weight

    # axis-centered interior
    gx[1:-1, 1:-1, 1:-1] = (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]) * inv_2dx
    gy[1:-1, 1:-1, 1:-1] = (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]) * inv_2dx
    gz[1:-1, 1:-1, 1:-1] = (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]) * inv_2dx

    # xy-diagonals
    gx[1:-1, 1:-1, 1:-1] = (
        w_axis * gx[1:-1, 1:-1, 1:-1]
        + w_diag * (
            (field[2:, 2:, 1:-1] - field[:-2, :-2, 1:-1]) * inv_2sqrt2dx
            + (field[2:, :-2, 1:-1] - field[:-2, 2:, 1:-1]) * inv_2sqrt2dx
        )
    )
    gy[1:-1, 1:-1, 1:-1] = (
        w_axis * gy[1:-1, 1:-1, 1:-1]
        + w_diag * (
            (field[2:, 2:, 1:-1] - field[:-2, :-2, 1:-1]) * inv_2sqrt2dx
            + (field[:-2, 2:, 1:-1] - field[2:, :-2, 1:-1]) * inv_2sqrt2dx
        )
    )

    # xz-diagonals
    gx[1:-1, 1:-1, 1:-1] += w_diag * (
        (field[2:, 1:-1, 2:] - field[:-2, 1:-1, :-2]) * inv_2sqrt2dx
        + (field[2:, 1:-1, :-2] - field[:-2, 1:-1, 2:]) * inv_2sqrt2dx
    )
    gz[1:-1, 1:-1, 1:-1] = (
        w_axis * gz[1:-1, 1:-1, 1:-1]
        + w_diag * (
            (field[2:, 1:-1, 2:] - field[:-2, 1:-1, :-2]) * inv_2sqrt2dx
            + (field[:-2, 1:-1, 2:] - field[2:, 1:-1, :-2]) * inv_2sqrt2dx
        )
    )

    # yz-diagonals
    gy[1:-1, 1:-1, 1:-1] += w_diag * (
        (field[1:-1, 2:, 2:] - field[1:-1, :-2, :-2]) * inv_2sqrt2dx
        + (field[1:-1, 2:, :-2] - field[1:-1, :-2, 2:]) * inv_2sqrt2dx
    )
    gz[1:-1, 1:-1, 1:-1] += w_diag * (
        (field[1:-1, 2:, 2:] - field[1:-1, :-2, :-2]) * inv_2sqrt2dx
        + (field[1:-1, :-2, 2:] - field[1:-1, 2:, :-2]) * inv_2sqrt2dx
    )

    _zero_face_gradients(gx, gy, gz)
    return gx, gy, gz


@njit(cache=True, fastmath=True)
def _divergence_from_components(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float) -> np.ndarray:
    """
    Divergence via centered differences of the provided components.
    Faces use one-sided forms consistent with zero normal derivative.
    """
    div = np.zeros_like(ux)
    inv_2dx = 1.0 / (2.0 * dx)

    # interior
    div[1:-1, 1:-1, 1:-1] = (
        (ux[2:, 1:-1, 1:-1] - ux[:-2, 1:-1, 1:-1])
        + (uy[1:-1, 2:, 1:-1] - uy[1:-1, :-2, 1:-1])
        + (uz[1:-1, 1:-1, 2:] - uz[1:-1, 1:-1, :-2])
    ) * inv_2dx

    # x-faces
    div[0, 1:-1, 1:-1] = (
        (ux[1, 1:-1, 1:-1] - ux[0, 1:-1, 1:-1])
        + (uy[0, 2:, 1:-1] - uy[0, :-2, 1:-1])
        + (uz[0, 1:-1, 2:] - uz[0, 1:-1, :-2])
    ) * inv_2dx
    div[-1, 1:-1, 1:-1] = (
        (ux[-1, 1:-1, 1:-1] - ux[-2, 1:-1, 1:-1])
        + (uy[-1, 2:, 1:-1] - uy[-1, :-2, 1:-1])
        + (uz[-1, 1:-1, 2:] - uz[-1, 1:-1, :-2])
    ) * inv_2dx

    # y-faces
    div[1:-1, 0, 1:-1] = (
        (ux[2:, 0, 1:-1] - ux[:-2, 0, 1:-1])
        + (uy[1:-1, 1, 1:-1] - uy[1:-1, 0, 1:-1])
        + (uz[1:-1, 0, 2:] - uz[1:-1, 0, :-2])
    ) * inv_2dx
    div[1:-1, -1, 1:-1] = (
        (ux[2:, -1, 1:-1] - ux[:-2, -1, 1:-1])
        + (uy[1:-1, -1, 1:-1] - uy[1:-1, -2, 1:-1])
        + (uz[1:-1, -1, 2:] - uz[1:-1, -1, :-2])
    ) * inv_2dx

    # z-faces
    div[1:-1, 1:-1, 0] = (
        (ux[2:, 1:-1, 0] - ux[:-2, 1:-1, 0])
        + (uy[1:-1, 2:, 0] - uy[1:-1, :-2, 0])
        + (uz[1:-1, 1:-1, 1] - uz[1:-1, 1:-1, 0])
    ) * inv_2dx
    div[1:-1, 1:-1, -1] = (
        (ux[2:, 1:-1, -1] - ux[:-2, 1:-1, -1])
        + (uy[1:-1, 2:, -1] - uy[1:-1, :-2, -1])
        + (uz[1:-1, 1:-1, -1] - uz[1:-1, 1:-1, -2])
    ) * inv_2dx

    return div


@njit(cache=True, fastmath=True)
def isotropic_laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """∇²_iso φ := ∇·(isotropic ∇φ)."""
    gix, giy, giz = isotropic_gradient_components(field, dx)
    return _divergence_from_components(gix, giy, giz, dx)


@njit(cache=True, fastmath=True)
def isotropic_gradient(field: np.ndarray, dx: float) -> np.ndarray:
    """|∇φ| computed from isotropic gradient components."""
    gix, giy, giz = isotropic_gradient_components(field, dx)
    return np.sqrt(gix * gix + giy * giy + giz * giz)
