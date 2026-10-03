# CT Reconstruction Lab

An interactive web app that shows how a CT scanner works as **linear algebra**.

An X-ray scan measures sums of densities along lines. Stack all the line weights into a sparse matrix **A**,
the unknown image into a vector **x**, and the measurements into **b**. Reconstruction is the system **Ax = b**.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

Run the tests with `python -m pytest -q`.

## What is in the app

| Section | What it shows | Linear algebra concept |
|---|---|---|
| Reconstruct | True image, sinogram, reconstruction, error image | Ax = b, least squares |
| Matrix A | One ray = one row of A; non-zero pattern; sparse vs dense memory | Sparse matrices, Radon transform |
| Angles & rank | Reconstructions with 3, 5, 10 ... 60 angles; rank vs. angles | Rank, null space, under-determined systems |
| Singular values | Singular value plot, Picard plot, singular vectors as images | SVD, condition number |
| Noise & regularization | Least squares vs. truncated SVD vs. ridge as noise grows | Ill-posedness, regularisation, bias-variance |
| Null-space ghost | Two different images with identical scans | Null space, x = A⁺Ax + (I - A⁺A)x |
| Theory | The maths behind each part | |

Sidebar controls: phantom (or upload your own picture), image size (16 to 64), number of angles, noise level and seed,
reconstruction method (least squares, truncated SVD, ridge, LSQR), and its parameter.

## Project layout

| File | Contents | Suggested owner |
|---|---|---|
| `radon.py` | `build_matrix(n, angles, n_detectors)` returns the sparse A (exact ray/pixel lengths) | Matrix lead |
| `recon.py` | `pinv_recon`, `tsvd_recon`, `ridge_recon`, `lsqr_recon`, `rmse`, rank / condition number, null-space projection, parameter sweeps | Reconstruction lead |
| `app.py` | Streamlit UI, sidebar, sections, caching | UI lead |
| `plots.py` | Every matplotlib figure | Analysis + experiments |
| `phantoms.py` | Shepp-Logan, blocks, image upload | Anyone |
| `tests/test_core.py` | 12 tests for geometry, recovery, null space, regularisation | Report / GitHub lead |

Conventions: the image is flattened row by row (`index = row * n + col`); rows of A are ordered angle by angle,
so `(A @ x).reshape(n_angles, n_detectors)` is the sinogram.

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (the `requirements.txt` is already included).
2. Go to https://share.streamlit.io, sign in with GitHub, choose **New app**, select the repo and set the main file to `app.py`.
3. Click **Deploy**. You get a public link for your report.

## Notes

- The SVD is the expensive step. It is cached, so it only runs again when image size, number of angles or detector count change.
  Typical first-time cost: about 1 s for 32 × 32, 7 s for 48 × 48, and roughly 40 s for 64 × 64.
- Only the selected section computes, so moving a slider in one tab never triggers heavy work in another.
- In *Noise & regularization*, the best k and best λ are picked by comparing with the true image (an oracle) to show the best each
  method can achieve. Real systems use the L-curve or cross-validation.
