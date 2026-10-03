"""
recon.py -- solving  A x = b  for the image x.

Every function takes the measurements `b` (flat vector) and returns an (n, n) image.
Methods that need the SVD take the factors (U, s, Vt) from `svd_decompose(A)` so the
expensive decomposition is computed once and reused by sliders.

Notation (A is m x N with N = n*n pixels, economy SVD):  A = U diag(s) Vt
"""
import numpy as np
from scipy.sparse.linalg import lsqr


# --------------------------------------------------------------------------- utilities
def rmse(x_true, x_rec):
    """Root-mean-square error = ||x_rec - x_true||_2 / sqrt(N)."""
    d = np.asarray(x_rec, dtype=float) - np.asarray(x_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def add_noise(b, noise_pct, seed=0):
    """Gaussian noise whose std is `noise_pct` percent of the largest measurement."""
    if noise_pct <= 0:
        return np.asarray(b, dtype=float).copy()
    rng = np.random.default_rng(seed)
    sigma = noise_pct / 100.0 * np.max(np.abs(b))
    return b + rng.normal(0.0, sigma, size=np.shape(b))


# --------------------------------------------------------------------------- SVD tools
def svd_decompose(A):
    """Economy SVD of the (densified) projection matrix: returns U, s, Vt."""
    dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    U, s, Vt = np.linalg.svd(dense, full_matrices=False)
    return U, s, Vt


def numerical_rank(s, shape=None):
    """Number of singular values above the usual floating-point threshold."""
    if s.size == 0:
        return 0
    m, N = shape if shape is not None else (s.size, s.size)
    tol = s.max() * max(m, N) * np.finfo(float).eps
    return int(np.sum(s > tol))


def condition_number(s, shape=None):
    """sigma_max / sigma_min over the non-zero singular values (rank-aware)."""
    r = numerical_rank(s, shape)
    return float(s[0] / s[r - 1]) if r > 0 else np.inf


def _apply_filter(U, s, Vt, b, filt, n):
    """x = V diag(filt / s) U^T b  -- the general 'filtered SVD' solution."""
    coeff = U.T @ np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(filt > 0, filt / s, 0.0)
    return (Vt.T @ (inv * coeff)).reshape(n, n)


def pinv_recon(U, s, Vt, b, n, shape=None):
    """Plain least squares: minimum-norm solution x = A^+ b (all non-zero sigma kept)."""
    r = numerical_rank(s, shape)
    filt = np.zeros_like(s)
    filt[:r] = 1.0
    return _apply_filter(U, s, Vt, b, filt, n)


def tsvd_recon(U, s, Vt, b, k, n):
    """Truncated SVD: keep only the k largest singular values."""
    k = int(max(1, min(k, s.size)))
    filt = np.zeros_like(s)
    filt[:k] = 1.0
    return _apply_filter(U, s, Vt, b, filt, n)


def ridge_recon(U, s, Vt, b, lam, n):
    """
    Ridge / Tikhonov:  argmin ||Ax - b||^2 + lam ||x||^2.
    In the SVD basis this multiplies each component by the filter s^2 / (s^2 + lam).
    """
    filt = s**2 / (s**2 + lam)
    return _apply_filter(U, s, Vt, b, filt, n)


def lsqr_recon(A, b, n, iters=200):
    """Sparse iterative least squares (LSQR). Stopping early acts as regularisation."""
    x = lsqr(A, np.asarray(b, dtype=float), atol=1e-12, btol=1e-12, iter_lim=int(iters))[0]
    return x.reshape(n, n)


# --------------------------------------------------------------------------- analysis
def picard_coefficients(U, b):
    """|u_i^T b| for every left singular vector (the 'Picard plot' data)."""
    return np.abs(U.T @ np.asarray(b, dtype=float))


def row_space_projection(Vt, s, image, shape=None):
    """Part of the image the scanner CAN see: V_r V_r^T x (= A^+ A x)."""
    r = numerical_rank(s, shape)
    Vr = Vt[:r].T
    x = np.asarray(image, dtype=float).ravel()
    return (Vr @ (Vr.T @ x)).reshape(np.shape(image))


def null_space_component(Vt, s, image, shape=None):
    """Part of the image the scanner is BLIND to: x - A^+ A x. Satisfies A g = 0."""
    return np.asarray(image, dtype=float) - row_space_projection(Vt, s, image, shape)


def _sweep_errors(U, s, Vt, b, x_true, filters, shape):
    """RMSE of many filtered-SVD solutions at once (one matrix product, no loops)."""
    r = numerical_rank(s, shape)
    inv = np.zeros_like(s)
    inv[:r] = 1.0 / s[:r]
    coeff = (U.T @ np.asarray(b, dtype=float)) * inv
    X = (filters * coeff[None, :]) @ Vt  # (n_params, N) -- each row is one reconstruction
    diff = X - np.asarray(x_true, dtype=float).ravel()[None, :]
    return np.sqrt(np.mean(diff**2, axis=1))


def tsvd_sweep(U, s, Vt, b, x_true, shape=None, ks=None):
    """RMSE for many TSVD cut-offs k (an 'oracle' sweep: it uses the true image)."""
    if ks is None:
        r = numerical_rank(s, shape)
        ks = np.unique(np.linspace(1, r, 60).astype(int))
    ks = np.asarray(ks)
    filters = (np.arange(s.size)[None, :] < ks[:, None]).astype(float)
    return ks, _sweep_errors(U, s, Vt, b, x_true, filters, shape)


def ridge_sweep(U, s, Vt, b, x_true, shape=None, rels=None):
    """RMSE for many ridge parameters, expressed relative to sigma_1^2 (lam = rel * s1^2)."""
    if rels is None:
        rels = np.logspace(-9, 0, 46)
    rels = np.asarray(rels)
    lams = rels * s[0] ** 2
    filters = s[None, :] ** 2 / (s[None, :] ** 2 + lams[:, None])
    return rels, _sweep_errors(U, s, Vt, b, x_true, filters, shape)
