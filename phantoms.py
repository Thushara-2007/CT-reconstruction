"""
phantoms.py -- test images (densities in [0, 1]) for the CT demo.

All images are returned as float arrays of shape (n, n); row 0 is the top.
No external data files are needed: the Shepp-Logan head phantom is generated
from its ten ellipses.
"""
import numpy as np

# (additive intensity, semi-axis a, semi-axis b, centre x0, centre y0, rotation in degrees)
# "Modified" Shepp-Logan parameters (higher contrast than the 1974 original).
_SHEPP_LOGAN = [
    (1.00, 0.6900, 0.9200, 0.00, 0.0000, 0),
    (-0.80, 0.6624, 0.8740, 0.00, -0.0184, 0),
    (-0.20, 0.1100, 0.3100, 0.22, 0.0000, -18),
    (-0.20, 0.1600, 0.4100, -0.22, 0.0000, 18),
    (0.10, 0.2100, 0.2500, 0.00, 0.3500, 0),
    (0.10, 0.0460, 0.0460, 0.00, 0.1000, 0),
    (0.10, 0.0460, 0.0460, 0.00, -0.1000, 0),
    (0.10, 0.0460, 0.0880, -0.08, -0.6050, 0),
    (0.10, 0.0230, 0.0230, 0.00, -0.6060, 0),
    (0.10, 0.0230, 0.0460, 0.06, -0.6050, 0),
]


def _grid(n):
    """Pixel-centre coordinates in [-1, 1]; y points up, row 0 is the top."""
    xs = (np.arange(n) + 0.5) / n * 2 - 1
    ys = 1 - (np.arange(n) + 0.5) / n * 2
    return np.meshgrid(xs, ys)


def shepp_logan(n):
    X, Y = _grid(n)
    img = np.zeros((n, n))
    for amp, a, b, x0, y0, phi in _SHEPP_LOGAN:
        p = np.deg2rad(phi)
        xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
        yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += amp
    img = np.clip(img, 0.0, None)
    return img / img.max()


def blocks(n):
    """Simple shapes: a disc, two rectangles and a small bright square."""
    X, Y = _grid(n)
    img = np.zeros((n, n))
    img[X**2 + Y**2 <= 0.85**2] = 0.25
    img[(np.abs(X + 0.35) < 0.25) & (np.abs(Y - 0.30) < 0.15)] = 0.7
    img[(np.abs(X - 0.30) < 0.12) & (np.abs(Y + 0.25) < 0.35)] = 0.55
    img[(X + 0.40) ** 2 + (Y + 0.40) ** 2 <= 0.18**2] = 1.0
    img[(np.abs(X - 0.35) < 0.07) & (np.abs(Y - 0.45) < 0.07)] = 0.9
    return img


def from_image(file_or_path, n):
    """Load any picture, convert to grayscale, centre-crop to a square, resize to n x n."""
    from PIL import Image

    im = Image.open(file_or_path).convert("L")
    w, h = im.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    im = im.crop((left, top, left + m, top + m)).resize((n, n), Image.LANCZOS)
    arr = np.asarray(im, dtype=float)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr


PHANTOMS = {
    "Shepp-Logan head": shepp_logan,
    "Blocks": blocks,
}


def get_phantom(name, n):
    return PHANTOMS[name](n)
