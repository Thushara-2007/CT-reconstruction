"""
radon.py -- the discrete Radon transform as a sparse matrix A.

Geometry
--------
* The image is an n x n grid of unit-size pixels centred on the origin, so it
  occupies the square [-n/2, n/2] x [-n/2, n/2].
* Pixel (row i, column j) has flat index  i*n + j.  Row 0 is the TOP of the image.
* A ray is the line  p(t) = s*(-sin(theta), cos(theta)) + t*(cos(theta), sin(theta)),
  i.e. a line at angle `theta` whose signed distance from the centre is `s`.
* The weight A[ray, pixel] is the exact LENGTH of the ray inside that pixel
  (a Siddon-style ray tracer: find where the ray crosses every vertical and
  horizontal grid line, sort the crossings, and every piece between two
  consecutive crossings lies inside exactly one pixel).

Row ordering of A is angle-major:  row = angle_index * n_detectors + detector_index
so   (A @ x).reshape(n_angles, n_detectors)   is the sinogram.
"""
import numpy as np
from scipy import sparse

_EPS = 1e-12


def make_angles(n_angles, span=180.0):
    """Evenly spaced projection angles in degrees on [0, span)."""
    return np.linspace(0.0, span, int(n_angles), endpoint=False)


def default_n_detectors(n):
    """A detector row slightly wider than the image diagonal sampling."""
    return int(np.ceil(1.5 * n))


def detector_positions(n, n_detectors):
    """Signed offsets s of the detector bins; they cover the image diagonal."""
    smax = n * np.sqrt(2.0) / 2.0
    ds = 2.0 * smax / n_detectors
    return -smax + (np.arange(n_detectors) + 0.5) * ds


def _empty():
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)


def _clip_to_box(n, theta, s):
    """Return (t0, t1, px, py, c, sn) for the part of the ray inside the image box."""
    half = n / 2.0
    c, sn = np.cos(theta), np.sin(theta)
    px, py = -s * sn, s * c  # point of the line closest to the origin
    t0, t1 = -np.inf, np.inf
    for p, d in ((px, c), (py, sn)):
        if abs(d) < _EPS:  # ray parallel to this axis
            if p < -half or p > half:
                return None
        else:
            ta, tb = (-half - p) / d, (half - p) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
    if t1 - t0 <= _EPS:
        return None
    return t0, t1, px, py, c, sn


def ray_pixels(n, theta, s):
    """Pixels crossed by one ray and the length of the ray in each of them."""
    clipped = _clip_to_box(n, theta, s)
    if clipped is None:
        return _empty()
    t0, t1, px, py, c, sn = clipped
    half = n / 2.0
    grid = np.arange(n + 1) - half  # positions of the grid lines
    parts = [np.array([t0, t1])]
    if abs(c) > _EPS:
        parts.append((grid - px) / c)  # crossings with vertical lines x = const
    if abs(sn) > _EPS:
        parts.append((grid - py) / sn)  # crossings with horizontal lines y = const
    t = np.concatenate(parts)
    t = np.sort(t[(t >= t0) & (t <= t1)])
    dt = np.diff(t)
    keep = dt > _EPS
    if not keep.any():
        return _empty()
    mid = (0.5 * (t[:-1] + t[1:]))[keep]
    dt = dt[keep]
    x, y = px + mid * c, py + mid * sn
    col = np.clip(np.floor(x + half).astype(np.int64), 0, n - 1)
    row = np.clip(np.floor(half - y).astype(np.int64), 0, n - 1)
    return row * n + col, dt


def build_matrix(n, angles_deg, n_detectors=None):
    """
    Sparse projection matrix A of shape (n_angles * n_detectors, n*n), CSR format.
    A[r, p] = length of ray r inside pixel p.
    """
    if n_detectors is None:
        n_detectors = default_n_detectors(n)
    thetas = np.deg2rad(np.asarray(angles_deg, dtype=float))
    offsets = detector_positions(n, n_detectors)

    rows, cols, vals = [], [], []
    r = 0
    for th in thetas:
        for s in offsets:
            idx, ln = ray_pixels(n, th, s)
            if idx.size:
                rows.append(np.full(idx.size, r, dtype=np.int64))
                cols.append(idx)
                vals.append(ln)
            r += 1
    if rows:
        data = (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)))
    else:
        data = (np.empty(0), (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
    return sparse.coo_matrix(data, shape=(r, n * n)).tocsr()


def forward(A, image, n_angles):
    """Simulate the scan: return the sinogram (n_angles, n_detectors)."""
    b = A @ np.asarray(image, dtype=float).ravel()
    return b.reshape(n_angles, -1)


def ray_endpoints(n, theta_deg, s):
    """End points ((x0, y0), (x1, y1)) of a ray inside the image box (for drawing)."""
    clipped = _clip_to_box(n, np.deg2rad(theta_deg), s)
    if clipped is None:
        return None
    t0, t1, px, py, c, sn = clipped
    return (px + t0 * c, py + t0 * sn), (px + t1 * c, py + t1 * sn)
