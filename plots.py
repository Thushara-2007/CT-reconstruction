"""
plots.py -- every matplotlib figure used by the app.

Kept out of app.py so the analysis / experiment people can reuse and tweak them.
Colour rules (reference palette, first three categorical slots are validated for
adjacent AND all-pairs colour-blind separation):
    blue = least squares / first series, orange = truncated SVD, aqua = ridge.
Signed quantities (errors, null-space images) use a blue - gray - red diverging map;
magnitudes use a one-hue blue ramp; images use plain grayscale.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import radon

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

DIV = LinearSegmentedColormap.from_list("ct_div", [BLUE, "#f0efec", RED])
SEQ = LinearSegmentedColormap.from_list(
    "ct_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
SEQ.set_under(SURFACE)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 9.5,
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
    }
)


# ------------------------------------------------------------------ small helpers
def _img(ax, img, title, vmin=0.0, vmax=1.0, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _sino(ax, sino, n, title, cmap="gray", vmin=None, vmax=None):
    smax = n * np.sqrt(2.0) / 2.0
    ax.imshow(
        sino.T,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=[0, 180, -smax, smax],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([0, 90, 180])
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("detector offset s")


def _lim(a):
    m = float(np.max(np.abs(a)))
    return m if m > 1e-12 else 1.0


# ------------------------------------------------------------------ reconstruct tab
def fig_overview(x_true, sino, rec, n, rmse_val, rec_label):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.7), gridspec_kw={"width_ratios": [1, 1.2, 1, 1]})
    _img(ax[0], x_true, r"True image  $x$")
    _sino(ax[1], sino, n, r"Sinogram  $b = Ax$ (+ noise)")
    _img(ax[2], rec, f"{rec_label}\nRMSE {rmse_val:.4f}")
    err = rec - x_true
    lim = _lim(err)
    _img(ax[3], err, f"Error  (colour range ±{lim:.2f})", -lim, lim, DIV)
    return fig


# ------------------------------------------------------------------ matrix tab
def fig_ray(A, x_true, n, angles_deg, a_idx, d_idx, nd):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    h = n / 2.0
    theta = float(angles_deg[a_idx])
    ax[0].imshow(x_true, cmap="gray", vmin=0, vmax=1, extent=[-h, h, -h, h], interpolation="nearest")
    for j, s in enumerate(radon.detector_positions(n, nd)):
        seg = radon.ray_endpoints(n, theta, s)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        if j == d_idx:
            ax[0].plot([x0, x1], [y0, y1], color=ORANGE, lw=2.4, zorder=3)
        else:
            ax[0].plot([x0, x1], [y0, y1], color=AXIS, lw=0.7, alpha=0.85, zorder=2)
    ax[0].set_xlim(-h, h)
    ax[0].set_ylim(-h, h)
    ax[0].set_title(f"All {nd} rays at {theta:.1f}°  (selected ray in orange)")
    ax[0].grid(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    row = a_idx * nd + d_idx
    w = A.getrow(row).toarray().reshape(n, n)
    top = max(float(w.max()), 1e-6)
    im = ax[1].imshow(w, cmap=SEQ, vmin=1e-9, vmax=top, interpolation="nearest")
    ax[1].set_title(f"Row {row} of A: {int((w > 0).sum())} non-zero weights")
    ax[1].grid(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    for sp in ax[1].spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.set_label("length of ray inside pixel")
    cb.outline.set_visible(False)
    return fig


def fig_sparsity(A, n_angles, nd):
    """Zoom on the first (up to) two angles: each row is one ray, each dot a pixel it crosses."""
    show_angles = min(2, n_angles)
    rows = show_angles * nd
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.spy(A[:rows], markersize=1.6, color=BLUE, aspect="auto")
    ax.grid(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("pixel index (columns of A)")
    ax.set_ylabel("ray index (rows of A)")
    ax.set_title(f"Non-zero pattern of A, first {show_angles} angle(s): {rows} of {A.shape[0]} rows, all {A.shape[1]} columns")
    return fig


# ------------------------------------------------------------------ angles tab
def fig_angles_grid(ks, recs, errs, n):
    cols = min(len(ks), 6)
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.75 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k, r, e in zip(axes.ravel(), ks, recs, errs):
        ax.axis("on")
        _img(ax, r, f"{k} angles\nRMSE {e:.3f}")
    return fig


def fig_angles_curves(ks, ranks, N, errs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    a1.plot(ks, ranks, color=BLUE, marker="o", ms=6)
    a1.axhline(N, color=MUTED, lw=1, ls="--")
    a1.text(ks[0], N, f"  N = {N} unknowns (full rank)", va="bottom", color=INK2)
    a1.set_ylim(0, N * 1.12)
    a1.set_xlabel("number of angles")
    a1.set_ylabel("rank of A")
    a1.set_title("Rank of A vs. number of angles")
    a2.plot(ks, errs, color=ORANGE, marker="o", ms=6)
    a2.set_xlabel("number of angles")
    a2.set_ylabel("RMSE")
    a2.set_title("Reconstruction error vs. number of angles")
    a2.set_ylim(bottom=0)
    return fig


# ------------------------------------------------------------------ singular values tab
def fig_singular(s, k_cut, cond):
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    idx = np.arange(1, s.size + 1)
    ax.semilogy(idx, s, color=BLUE)
    ax.axvline(k_cut, color=ORANGE, lw=1.6)
    ax.text(k_cut, s.max(), f"  cut-off k = {k_cut}", color=INK2, va="top", ha="left")
    ax.annotate(rf"$\sigma_1$ = {s[0]:.3g}", (1, s[0]), xytext=(8, -14), textcoords="offset points", color=INK2)
    ax.annotate(
        rf"$\sigma_N$ = {s[-1]:.3g}", (idx[-1], s[-1]), xytext=(-8, 10), textcoords="offset points", color=INK2, ha="right"
    )
    ax.set_xlabel("index i")
    ax.set_ylabel(r"singular value $\sigma_i$")
    ax.set_title(f"Singular values of A   (condition number ≈ {cond:,.0f})")
    return fig


def _smooth_log(y, w):
    """Running mean in the log domain (the raw coefficients jump around by orders of magnitude)."""
    w = max(3, int(w) | 1)  # odd window
    ly = np.log10(np.clip(y, 1e-300, None))
    padded = np.pad(ly, w // 2, mode="edge")
    return 10 ** np.convolve(padded, np.ones(w) / w, mode="valid")[: y.size]


def fig_picard(s, c_clean, c_noisy, k_cut):
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    idx = np.arange(1, s.size + 1)
    floor = s.min() * 0.3
    clean = np.clip(c_clean, floor, None)
    noisy = np.clip(c_noisy, floor, None)
    w = max(5, s.size // 40)
    ax.semilogy(idx, clean, color=AQUA, lw=0.7, alpha=0.3)
    ax.semilogy(idx, noisy, color=ORANGE, lw=0.7, alpha=0.3)
    ax.semilogy(idx, _smooth_log(clean, w), color=AQUA, label="clean data (smoothed)")
    ax.semilogy(idx, _smooth_log(noisy, w), color=ORANGE, label="noisy data (smoothed)")
    ax.semilogy(idx, s, color=BLUE, label=r"singular value $\sigma_i$")
    ax.axvline(k_cut, color=INK2, lw=1.0, ls=":")
    top = max(c_noisy.max(), c_clean.max(), s.max())
    ax.set_ylim(floor, top * 12)  # headroom for the legend
    ax.set_xlabel("index i")
    ax.set_ylabel(r"magnitude of $|u_i^T b|$")
    ax.set_title("Picard plot: signal vs. noise per component")
    ax.legend(loc="upper center", ncol=3, fontsize=8, columnspacing=1.0, handlelength=1.4)
    return fig


def fig_modes(Vt, s, idxs, n):
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.3 * len(idxs), 2.8), squeeze=False)
    for ax, i in zip(axes[0], idxs):
        v = Vt[i].reshape(n, n)
        lim = _lim(v)
        _img(ax, v, rf"$v_{{{i + 1}}}$   ($\sigma$ = {s[i]:.3g})", -lim, lim, DIV)
    return fig


# ------------------------------------------------------------------ noise tab
def fig_noise_images(x_true, items):
    """items: list of (title, image, rmse or None)."""
    fig, axes = plt.subplots(1, len(items) + 1, figsize=(3.2 * (len(items) + 1), 3.5))
    _img(axes[0], x_true, "True image")
    for ax, (title, img, e) in zip(axes[1:], items):
        _img(ax, img, f"{title}\nRMSE {e:.4f}")
    return fig


def fig_sweeps(ks, e_k, rels, e_l):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    ib = int(np.argmin(e_k))
    a1.plot(ks, e_k, color=ORANGE)
    a1.plot([ks[ib]], [e_k[ib]], "o", color=ORANGE, ms=8, mec=SURFACE, mew=2)
    a1.text(0.5, 0.95, f"best k = {ks[ib]},  RMSE {e_k[ib]:.4f}", transform=a1.transAxes, ha="center", va="top", color=INK2)
    a1.set_xlabel("components kept  k")
    a1.set_ylabel("RMSE")
    a1.set_title("Truncated SVD: too few = blurry, too many = noisy")
    ib = int(np.argmin(e_l))
    a2.semilogx(rels, e_l, color=AQUA)
    a2.plot([rels[ib]], [e_l[ib]], "o", color=AQUA, ms=8, mec=SURFACE, mew=2)
    a2.text(
        0.5,
        0.95,
        rf"best $\lambda/\sigma_1^2$ = {rels[ib]:.1e},  RMSE {e_l[ib]:.4f}",
        transform=a2.transAxes,
        ha="center",
        va="top",
        color=INK2,
    )
    a2.set_xlabel(r"regularisation  $\lambda / \sigma_1^2$")
    a2.set_ylabel("RMSE")
    a2.set_title("Ridge: too small = noisy, too large = blurry")
    return fig


def fig_noise_vs_method(levels, e_ls, e_ts, e_rd):
    fig, ax = plt.subplots(figsize=(8.2, 3.9), layout="none")  # manual margins leave room for the end labels
    ax.plot(levels, e_ls, color=BLUE, marker="o", ms=6, label="least squares")
    ax.plot(levels, e_ts, color=ORANGE, marker="o", ms=6, label="truncated SVD (best k)")
    ax.plot(levels, e_rd, color=AQUA, marker="o", ms=6, label="ridge (best λ)")
    for y, text, va in ((e_ls[-1], "least squares", "center"), (e_ts[-1], "truncated SVD", "bottom"), (e_rd[-1], "ridge", "top")):
        ax.annotate(
            text, (levels[-1], y), xytext=(8, 0), textcoords="offset points", va=va, color=INK2, annotation_clip=False
        )
    ax.set_xlabel("noise level (% of max measurement)")
    ax.set_ylabel("RMSE")
    ax.set_title("Regularisation matters more as noise grows")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.84, bottom=0.15, top=0.9)
    return fig


# ------------------------------------------------------------------ ghost tab
def fig_ghost(x_true, x_vis, g, x_alt, sino_true, sino_alt, n, alpha, max_diff):
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.8), gridspec_kw={"height_ratios": [1, 0.95]})
    lo = float(min(x_true.min(), x_alt.min()))
    hi = float(max(x_true.max(), x_alt.max()))
    _img(ax[0, 0], x_true, r"True image  $x$", lo, hi)
    _img(ax[0, 1], x_vis, r"Visible part  $A^{+}A\,x$", lo, hi)
    gl = _lim(g)
    _img(ax[0, 2], g, r"Invisible part  $g$  ($Ag = 0$)", -gl, gl, DIV)
    _img(ax[0, 3], x_alt, rf"$x {alpha:+g}\,g$  (different image)", lo, hi)
    sm = max(float(sino_true.max()), 1e-12)
    _sino(ax[1, 0], sino_true, n, r"Scan of  $x$", vmin=0, vmax=sm)
    _sino(ax[1, 1], sino_alt, n, rf"Scan of  $x {alpha:+g}\,g$", vmin=0, vmax=sm)
    _sino(ax[1, 2], sino_alt - sino_true, n, f"Difference  (max {max_diff:.1e})", cmap=DIV, vmin=-0.01 * sm, vmax=0.01 * sm)
    ax[1, 3].axis("off")
    ax[1, 3].text(
        0.0,
        0.55,
        "Two different images,\nidentical measurements.\n\nThe scanner cannot tell\nthem apart.",
        transform=ax[1, 3].transAxes,
        fontsize=12,
        color=INK2,
        va="center",
    )
    return fig
