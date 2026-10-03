"""
app.py -- CT Reconstruction Lab (Streamlit).

Run with:   streamlit run app.py

An X-ray scan measures sums of densities along lines.  Stack all those sums into a
vector b, all the line weights into a sparse matrix A, and the unknown image into a
vector x: reconstruction is the linear system  A x = b.
"""
import io
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import phantoms
import plots
import radon
import recon

st.set_page_config(page_title="CT Reconstruction Lab", layout="wide")

SECTIONS = [
    "Reconstruct",
    "Matrix A",
    "Angles & rank",
    "Singular values",
    "Noise & regularization",
    "Null-space ghost",
    "Theory",
]
M_LS = "Least squares (pseudoinverse)"
M_TSVD = "Truncated SVD"
M_RIDGE = "Ridge (Tikhonov)"
M_LSQR = "LSQR (sparse, iterative)"


# --------------------------------------------------------------------------- caching
@st.cache_resource(max_entries=6, show_spinner=False)
def get_A(n, n_angles, nd):
    """Sparse projection matrix (cached so sliders never rebuild it needlessly)."""
    return radon.build_matrix(n, radon.make_angles(n_angles), nd)


@st.cache_resource(max_entries=3, show_spinner=False)
def get_svd(n, n_angles, nd):
    """Economy SVD of A (U, s, Vt) -- the expensive step, computed once per setup."""
    return recon.svd_decompose(get_A(n, n_angles, nd))


@st.cache_data(show_spinner=False)
def load_phantom(name, n, data):
    if data is not None:
        return phantoms.from_image(io.BytesIO(data), n)
    return phantoms.get_phantom(name, n)


@st.cache_data(show_spinner=False, max_entries=32)
def angle_study_item(n, k, nd, image, noise_pct, seed):
    """Plain least squares (np.linalg.lstsq) for one angle count."""
    A = get_A(n, k, nd)
    b = recon.add_noise(A @ image.ravel(), noise_pct, seed)
    x, _, rank, sv = np.linalg.lstsq(A.toarray(), b, rcond=None)
    rank = int(rank)
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else float("inf")
    rec = x.reshape(n, n)
    return {"k": k, "rows": A.shape[0], "rank": rank, "cond": cond, "rec": rec, "rmse": recon.rmse(image, rec)}


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- sidebar
def build_context():
    sb = st.sidebar
    sb.title("Scan settings")

    name = sb.selectbox("Image (phantom)", ["Shepp-Logan head", "Blocks", "Upload your own"])
    data = None
    if name == "Upload your own":
        up = sb.file_uploader("Image file", type=["png", "jpg", "jpeg", "bmp"])
        if up is not None:
            data = up.getvalue()
        else:
            sb.info("No file yet - showing the Shepp-Logan phantom.")
            name = "Shepp-Logan head"

    n = sb.select_slider("Image size  (n × n pixels)", options=[16, 24, 32, 48, 64], value=32)
    if n == 64:
        sb.warning("64 × 64 works, but the first SVD takes about a minute (it is cached afterwards).")
    n_angles = sb.slider("Number of angles (0° to 180°)", 1, 120, 40, help="Fewer angles = fewer equations.")
    noise_pct = sb.slider("Noise (% of max measurement)", 0.0, 20.0, 1.0, 0.5)
    seed = int(sb.number_input("Noise seed", min_value=0, value=0, step=1))

    sb.title("Reconstruction")
    method = sb.selectbox("Method", [M_LS, M_TSVD, M_RIDGE, M_LSQR])
    keep_pct, rel_lam, lsqr_iters = 70, 1e-3, 60
    if method == M_TSVD:
        keep_pct = sb.slider("Keep the top ... % of singular values", 1, 100, 70)
    elif method == M_RIDGE:
        e = sb.slider(r"log10( lambda / sigma_1^2 )", -8.0, 0.0, -3.0, 0.5)
        rel_lam = 10.0**e
    elif method == M_LSQR:
        lsqr_iters = sb.slider("Iterations", 1, 300, 60, help="Stopping early regularises the solution.")
    with sb.expander("Advanced"):
        nd = sb.slider("Detector bins", n, 3 * n, radon.default_n_detectors(n))

    x_true = load_phantom(name, n, data)
    A = get_A(n, n_angles, nd)
    b_clean = A @ x_true.ravel()
    b = recon.add_noise(b_clean, noise_pct, seed)
    return SimpleNamespace(
        n=n,
        n_angles=n_angles,
        nd=nd,
        noise_pct=noise_pct,
        seed=seed,
        method=method,
        keep_pct=keep_pct,
        rel_lam=rel_lam,
        lsqr_iters=lsqr_iters,
        x_true=x_true,
        A=A,
        b_clean=b_clean,
        b=b,
        angles=radon.make_angles(n_angles),
    )


def spin_svd(c, n_angles=None):
    with st.spinner("Computing the SVD of A (cached afterwards) ..."):
        return get_svd(c.n, c.n_angles if n_angles is None else n_angles, c.nd)


# --------------------------------------------------------------------------- sections
def section_reconstruct(c):
    n, N = c.n, c.n**2
    need_svd = c.method != M_LSQR or n <= 48
    svd = spin_svd(c) if need_svd else None

    if c.method == M_LSQR:
        rec = recon.lsqr_recon(c.A, c.b, n, c.lsqr_iters)
        label = f"LSQR, {c.lsqr_iters} iterations"
    else:
        U, s, Vt = svd
        rank = recon.numerical_rank(s, c.A.shape)
        if c.method == M_LS:
            rec, label = recon.pinv_recon(U, s, Vt, c.b, n, c.A.shape), "Least squares  (x = A⁺b)"
        elif c.method == M_TSVD:
            k = max(1, int(round(c.keep_pct / 100 * rank)))
            rec, label = recon.tsvd_recon(U, s, Vt, c.b, k, n), f"Truncated SVD, k = {k}"
        else:
            rec = recon.ridge_recon(U, s, Vt, c.b, c.rel_lam * s[0] ** 2, n)
            label = f"Ridge, λ/σ₁² = {c.rel_lam:.0e}"

    sino = c.b.reshape(c.n_angles, c.nd)
    e = recon.rmse(c.x_true, rec)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rays (rows of A)", f"{c.A.shape[0]:,}")
    m2.metric("Unknown pixels", f"{N:,}")
    if svd is not None:
        s = svd[1]
        rank = recon.numerical_rank(s, c.A.shape)
        m3.metric("Rank of A", f"{rank:,}")
        m4.metric("Condition number", f"{recon.condition_number(s, c.A.shape):,.0f}")
    else:
        m3.metric("Rank of A", "-")
        m4.metric("Condition number", "-")
    m5.metric("RMSE", f"{e:.4f}")

    show(plots.fig_overview(c.x_true, sino, rec, n, e, label))

    if svd is not None:
        if rank < N:
            st.warning(
                f"**A is rank-deficient** (rank {rank} < {N} unknowns). The null space has dimension {N - rank}: many different "
                "images produce exactly these measurements, and the solver can only return one of them (the minimum-norm one). "
                "Increase the number of angles to fix this."
            )
        elif recon.condition_number(s, c.A.shape) > 100 and c.noise_pct > 0:
            st.info(
                "A has full rank, but its smallest singular values are small, so noise in b gets amplified in x. "
                "Try *Truncated SVD* or *Ridge* in the sidebar and compare the RMSE."
            )
    with st.expander("How to read this"):
        st.markdown(
            "- **True image** is the unknown **x**; **sinogram** is the measured vector **b** drawn as an angle × detector picture.\n"
            "- **Reconstruction** is the solution of **Ax = b** found by the chosen method.\n"
            "- **Error** is reconstruction − truth: blue = too dark, red = too bright, gray = correct.\n"
            "- Use the sidebar: fewer angles make A rank-deficient (blurry streaks), more noise makes small singular values dangerous."
        )


def section_matrix(c):
    A = c.A
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    dense_mb = A.shape[0] * A.shape[1] * 8 / 1e6
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shape of A", f"{A.shape[0]:,} × {A.shape[1]:,}")
    m2.metric("Non-zeros", f"{A.nnz:,}", f"{100 * A.nnz / (A.shape[0] * A.shape[1]):.2f}% of entries", delta_color="off")
    m3.metric("Sparse storage", f"{sparse_mb:.2f} MB")
    m4.metric("Dense storage", f"{dense_mb:.1f} MB", f"{dense_mb / sparse_mb:.0f}× larger", delta_color="off")

    st.markdown(
        "Each **row** of A is one X-ray. Its entries are the length of the ray inside each pixel, "
        "so the measurement is the dot product *(row of A) · x*. Pick a ray below."
    )
    c1, c2 = st.columns(2)
    a_idx = c1.slider("Projection angle (index)", 0, max(c.n_angles - 1, 0), 0) if c.n_angles > 1 else 0
    d_idx = c2.slider("Detector bin", 0, c.nd - 1, c.nd // 2)
    show(plots.fig_ray(A, c.x_true, c.n, c.angles, a_idx, d_idx, c.nd))
    show(plots.fig_sparsity(A, c.n_angles, c.nd))
    st.caption(
        "Almost every entry is zero: a ray only touches about n of the n² pixels. Sparse storage keeps just the non-zeros. "
        "Each block of rows is one angle; within it the dots shift steadily as the detector moves across the image."
    )


def section_angles(c):
    st.markdown(
        "How many angles do we need? Below, each reconstruction is **plain least squares** "
        "(`np.linalg.lstsq`). With too few angles the matrix has **rank < N**: a null space exists and the answer is ambiguous."
    )
    options = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 90, 120]
    sel = st.multiselect("Angle counts to compare", options, default=[3, 5, 10, 20, 40, 60])
    use_noise = st.toggle("Add the sidebar noise level", value=False)
    if not sel:
        st.info("Pick at least one angle count.")
        return
    noise = c.noise_pct if use_noise else 0.0
    with st.spinner("Solving least squares for each angle count (cached afterwards) ..."):
        items = [angle_study_item(c.n, k, c.nd, c.x_true, noise, c.seed) for k in sorted(sel)]
    ks = [it["k"] for it in items]
    N = c.n**2
    show(plots.fig_angles_grid(ks, [it["rec"] for it in items], [it["rmse"] for it in items], c.n))
    show(plots.fig_angles_curves(ks, [it["rank"] for it in items], N, [it["rmse"] for it in items]))
    df = pd.DataFrame(
        {
            "angles": ks,
            "rays (rows)": [it["rows"] for it in items],
            "unknowns N": N,
            "rank": [it["rank"] for it in items],
            "null-space dim (N - rank)": [N - it["rank"] for it in items],
            "condition number": [round(it["cond"], 1) for it in items],
            "RMSE": [round(it["rmse"], 4) for it in items],
        }
    )
    with st.expander("Data table"):
        st.dataframe(df, hide_index=True)


def section_singular(c):
    U, s, Vt = spin_svd(c)
    rank = recon.numerical_rank(s, c.A.shape)
    cond = recon.condition_number(s, c.A.shape)
    c1, c2 = st.columns(2)
    keep = c1.slider("Cut-off to mark (% of singular values kept)", 1, 100, 70)
    noise = c2.slider("Noise for the Picard plot (%)", 0.0, 20.0, 1.0, 0.5)
    k_cut = max(1, int(round(keep / 100 * rank)))
    bn = recon.add_noise(c.b_clean, noise, c.seed)

    left, right = st.columns(2)
    with left:
        show(plots.fig_singular(s, k_cut, cond))
        st.caption("Reconstruction divides by these numbers. The tiny ones at the end are what makes the problem ill-conditioned.")
    with right:
        show(plots.fig_picard(s, recon.picard_coefficients(U, c.b_clean), recon.picard_coefficients(U, bn), k_cut))
        st.caption(
            "Noise adds a roughly flat floor to every coefficient uᵢᵀb. Once the clean signal (aqua) drops toward that floor, "
            "the component is mostly noise, and dividing it by a small σᵢ amplifies it. That is where truncation helps."
        )

    st.subheader("What the singular vectors look like")
    idxs = sorted({0, rank // 30, rank // 10, rank // 4, rank // 2, rank - 1})
    show(plots.fig_modes(Vt, s, idxs, c.n))
    st.caption("Large σ = smooth, coarse patterns that the scan measures well. Small σ = fine, oscillating patterns that barely show up in b.")


def section_noise(c):
    noise = st.slider("Noise level for this experiment (%)", 0.0, 20.0, 3.0, 0.5)
    U, s, Vt = spin_svd(c)
    shape, n = c.A.shape, c.n
    bn = recon.add_noise(c.b_clean, noise, c.seed)

    ks, e_k = recon.tsvd_sweep(U, s, Vt, bn, c.x_true, shape)
    rels, e_l = recon.ridge_sweep(U, s, Vt, bn, c.x_true, shape)
    kb, rb = int(ks[np.argmin(e_k)]), float(rels[np.argmin(e_l)])
    ls = recon.pinv_recon(U, s, Vt, bn, n, shape)
    ts = recon.tsvd_recon(U, s, Vt, bn, kb, n)
    rd = recon.ridge_recon(U, s, Vt, bn, rb * s[0] ** 2, n)
    show(
        plots.fig_noise_images(
            c.x_true,
            [
                ("Least squares", ls, recon.rmse(c.x_true, ls)),
                (f"Truncated SVD (k = {kb})", ts, recon.rmse(c.x_true, ts)),
                ("Ridge (best λ)", rd, recon.rmse(c.x_true, rd)),
            ],
        )
    )
    st.caption(
        "Parameters here are picked by comparing with the true image (an *oracle*), to show the best each method can do. "
        "Real scanners use tools like the L-curve or cross-validation."
    )
    show(plots.fig_sweeps(ks, e_k, rels, e_l))

    levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    e_ls, e_ts, e_rd = [], [], []
    for lv in levels:
        b2 = recon.add_noise(c.b_clean, lv, c.seed)
        e_ls.append(recon.rmse(c.x_true, recon.pinv_recon(U, s, Vt, b2, n, shape)))
        e_ts.append(float(recon.tsvd_sweep(U, s, Vt, b2, c.x_true, shape)[1].min()))
        e_rd.append(float(recon.ridge_sweep(U, s, Vt, b2, c.x_true, shape)[1].min()))
    show(plots.fig_noise_vs_method(levels, e_ls, e_ts, e_rd))
    with st.expander("Data table"):
        st.dataframe(
            pd.DataFrame(
                {"noise %": levels, "least squares": np.round(e_ls, 4), "truncated SVD": np.round(e_ts, 4), "ridge": np.round(e_rd, 4)}
            ),
            hide_index=True,
        )


def section_ghost(c):
    st.markdown(
        "If A has a **null space**, some images are completely invisible to the scanner. "
        "Split the true image as *x = (visible part) + g*, where **A g = 0**. "
        "Adding any multiple of g gives a different image with **exactly the same measurements**."
    )
    c1, c2 = st.columns(2)
    k = c1.slider("Number of angles", 2, 40, 6)
    alpha = c2.slider("Ghost amount α  (x + α·g)", -3.0, 3.0, 2.0, 0.25)
    A = get_A(c.n, k, c.nd)
    U, s, Vt = spin_svd(c, k)
    shape = A.shape
    rank = recon.numerical_rank(s, shape)
    N = c.n**2

    x = c.x_true
    g = recon.null_space_component(Vt, s, x, shape)
    if np.abs(g).max() < 1e-9:
        st.warning(f"With {k} angles A has full column rank ({rank} = N), so the null space is {{0}} and no ghost exists. Lower the number of angles.")
        return
    x_vis = x - g
    x_alt = x + alpha * g
    b_true = A @ x.ravel()
    b_alt = A @ x_alt.ravel()
    max_diff = float(np.max(np.abs(b_alt - b_true)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rank of A", f"{rank:,}")
    m2.metric("Null-space dimension", f"{N - rank:,}")
    m3.metric("‖x_alt − x‖ (pixel units)", f"{np.linalg.norm(x_alt - x):.2f}")
    m4.metric("max |scan difference|", f"{max_diff:.1e}")
    show(plots.fig_ghost(x, x_vis, g, x_alt, b_true.reshape(k, c.nd), b_alt.reshape(k, c.nd), c.n, alpha, max_diff))
    st.caption(
        "The *visible part* is also exactly what the pseudoinverse returns from noise-free data (the minimum-norm solution) - "
        "least squares can never recover the invisible part g."
    )


def section_theory(_c):
    st.header("The mathematics behind the app")
    st.markdown("### 1. The model: a linear system")
    st.markdown(
        "Flatten the $n\\times n$ image into a vector $x\\in\\mathbb{R}^{N}$, $N=n^2$. Ray $i$ measures the weighted sum of the "
        "pixels it crosses, with weights $A_{ij}$ = length of ray $i$ inside pixel $j$:"
    )
    st.latex(r"b_i=\sum_{j=1}^{N}A_{ij}\,x_j \quad\Longleftrightarrow\quad A\,x=b")
    st.markdown(
        "The continuous version of $A$ is the **Radon transform**; our matrix is its discretisation. "
        "$A$ is **sparse** because a ray touches only about $n$ of the $n^2$ pixels."
    )
    st.markdown("### 2. Rank and null space")
    st.markdown(
        "The **rank** is the number of independent measurements. The **null space** $\\{g : Ag=0\\}$ contains images the scanner cannot see, "
        "so $x$ and $x+g$ give identical data. Few angles make the rank drop and the null space grow."
    )
    st.latex(r"A(x+g)=Ax+Ag=Ax=b")
    st.markdown("### 3. Least squares and the pseudoinverse")
    st.markdown("With noise, $b$ is not in the column space of $A$, so we minimise the residual. The solution satisfies the normal equations:")
    st.latex(r"\min_x\|Ax-b\|_2^2 \;\Longrightarrow\; A^{T}A\,x=A^{T}b,\qquad x=A^{+}b")
    st.markdown("$A^{+}$ gives the **minimum-norm** least-squares solution when the null space is non-trivial.")
    st.markdown("### 4. SVD and the condition number")
    st.latex(r"A=U\Sigma V^{T},\qquad x=A^{+}b=\sum_{i=1}^{r}\frac{u_i^{T}b}{\sigma_i}\,v_i")
    st.markdown(
        "Each component is divided by $\\sigma_i$. If $b$ contains noise $\\varepsilon$, the error in $x$ contains "
        "$u_i^T\\varepsilon/\\sigma_i$: **tiny singular values amplify noise**. The condition number measures this:"
    )
    st.latex(r"\kappa(A)=\frac{\sigma_{\max}}{\sigma_{\min}}")
    st.markdown("### 5. Regularisation")
    st.markdown("Trade a little bias for a lot less noise by filtering the components:")
    st.latex(
        r"x_{\text{TSVD}}=\sum_{i=1}^{k}\frac{u_i^Tb}{\sigma_i}v_i,\qquad "
        r"x_{\text{ridge}}=\sum_{i}\frac{\sigma_i}{\sigma_i^2+\lambda}(u_i^Tb)\,v_i"
    )
    st.markdown(
        "Ridge (Tikhonov) solves $\\min\\|Ax-b\\|^2+\\lambda\\|x\\|^2$. LSQR stopped early acts as a third form of regularisation."
    )
    st.markdown("### 6. Error measure")
    st.latex(r"\mathrm{RMSE}=\frac{\|x_{\text{rec}}-x_{\text{true}}\|_2}{\sqrt{N}}")
    st.markdown("### Where each idea appears")
    st.markdown(
        "| App section | Concept |\n|---|---|\n"
        "| Matrix A | sparse matrices, ray/pixel geometry (Radon transform) |\n"
        "| Angles & rank | rank, null space, under-determined systems |\n"
        "| Singular values | SVD, condition number, Picard plot |\n"
        "| Noise & regularization | ill-posedness, truncated SVD, ridge, bias-variance trade-off |\n"
        "| Null-space ghost | null space, orthogonal decomposition $x=A^{+}Ax+(I-A^{+}A)x$ |\n"
    )
    st.markdown(
        "**Bonus:** the *Fourier slice theorem* says the 1-D Fourier transform of a projection equals a slice of the image's 2-D Fourier "
        "transform. Real CT scanners use it through *filtered back-projection*."
    )


DISPATCH = {
    "Reconstruct": section_reconstruct,
    "Matrix A": section_matrix,
    "Angles & rank": section_angles,
    "Singular values": section_singular,
    "Noise & regularization": section_noise,
    "Null-space ghost": section_ghost,
    "Theory": section_theory,
}


def main():
    ctx = build_context()
    st.title("CT Reconstruction Lab")
    st.caption(
        "An X-ray scan measures sums of densities along lines. Many sums from many angles form a linear system **A x = b**; "
        "solving it recovers the image."
    )
    section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="section", label_visibility="collapsed")
    DISPATCH[section or SECTIONS[0]](ctx)


main()
