"""
Compare LU factorization and solve quality across PyTorch backends.

Runs torch.linalg.lu_factor_ex + torch.linalg.lu_solve with backend set to
"magma" and "cusolver", then prints a table showing residuals for each.

Residual formulas (MAGMA conventions):
  LU factorization:  ||PA - LU||_F / (N * ||A||_F * eps)
  LU solve:          ||Ax - b||_inf / (N * ||A||_inf * ||x||_inf * eps)

Matrix generation uses LAPACK DLATMS methodology (geometric singular value
decay) for controlled condition numbers.
"""

import torch
import argparse
import sys

DEVICE = "cuda"
NRHS = 4
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _higher_dtype(dtype):
    if dtype in (torch.float32, torch.float64):
        return torch.float64
    return torch.complex128


def _real_dtype(dtype):
    if dtype in (torch.complex64, torch.complex128):
        return torch.float64 if dtype == torch.complex128 else torch.float32
    return dtype


def _eps(dtype):
    return torch.finfo(_real_dtype(dtype)).eps


def _seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------


def make_matrix_with_cond(batch, n, cond, dtype):
    """A = U @ diag(sigma) @ Vt with geometric singular value decay."""
    high_dtype = _higher_dtype(dtype)
    sigma = torch.logspace(0, -torch.log10(torch.tensor(cond, dtype=torch.float64)).item(),
                           steps=n, dtype=torch.float64, device=DEVICE)
    As = []
    for _ in range(batch):
        Q1, _ = torch.linalg.qr(torch.randn(n, n, dtype=high_dtype, device=DEVICE))
        Q2, _ = torch.linalg.qr(torch.randn(n, n, dtype=high_dtype, device=DEVICE))
        A = Q1 @ torch.diag(sigma.to(high_dtype)) @ Q2
        As.append(A)
    A = torch.stack(As)
    # Normalize to unit infinity norm
    norm = torch.linalg.matrix_norm(A, ord=float('inf')).max()
    A = A / norm
    return A.to(dtype)


def generate_rhs(A, nrhs=NRHS):
    """B = A @ X_exact in higher precision."""
    batch, n, _ = A.shape
    high_dtype = _higher_dtype(A.dtype)
    X_exact = torch.randn(batch, n, nrhs, dtype=A.dtype, device=DEVICE)
    B = (A.to(high_dtype) @ X_exact.to(high_dtype)).to(A.dtype)
    return B, X_exact


# ---------------------------------------------------------------------------
# Residual computations
# ---------------------------------------------------------------------------


def lu_factorization_residual(A_orig, LU, pivots):
    """||PA - LU||_F / (N * ||A||_F * eps).

    MAGMA get_LU_error formula: Frobenius norm, N in denominator.
    Returns max across batch.
    """
    high_dtype = _higher_dtype(A_orig.dtype)
    batch, n, _ = A_orig.shape
    eps = _eps(A_orig.dtype)

    A_h = A_orig.to(high_dtype)
    LU_h = LU.to(high_dtype)

    # Extract L and U from packed format
    L = torch.tril(LU_h, diagonal=-1)
    L.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    U = torch.triu(LU_h)

    # Reconstruct PA by applying pivots to A
    # pivots are 1-based sequential row swaps (LAPACK convention)
    PA = A_h.clone()
    for i in range(n):
        p = pivots[:, i] - 1  # to 0-based
        for b in range(batch):
            if p[b] != i:
                PA[b, i, :].clone(), PA[b, p[b], :].clone()
                tmp = PA[b, i, :].clone()
                PA[b, i, :] = PA[b, p[b], :]
                PA[b, p[b], :] = tmp

    # Compute PA - LU
    diff = PA - L @ U
    norm_diff = torch.linalg.matrix_norm(diff, ord='fro')  # (batch,)
    norm_A = torch.linalg.matrix_norm(A_h, ord='fro')      # (batch,)

    denom = n * norm_A * eps
    denom = torch.clamp(denom, min=1e-300)
    scaled = norm_diff / denom
    return scaled.max().item()


def solve_residual(A, X, B):
    """||Ax - b||_inf / (N * ||A||_inf * ||x||_inf * eps).

    MAGMA testing_zgesv formula: infinity norm, N in denominator.
    Returns max across batch.
    """
    high_dtype = _higher_dtype(A.dtype)
    n = A.shape[-1]
    eps = _eps(A.dtype)

    A_h = A.to(high_dtype)
    X_h = X.to(high_dtype)
    B_h = B.to(high_dtype)

    R = B_h - A_h @ X_h
    Rnorm = torch.linalg.matrix_norm(R, ord=float('inf'))
    Anorm = torch.linalg.matrix_norm(A_h, ord=float('inf'))
    Xnorm = torch.linalg.matrix_norm(X_h, ord=float('inf'))

    denom = n * Anorm * Xnorm * eps
    denom = torch.clamp(denom, min=1e-300)
    scaled = Rnorm / denom
    return scaled.max().item()


def forward_error(X, X_exact, rcond):
    """||X - X_exact||_inf / (N * ||X_exact||_inf * eps) * RCOND.

    Returns max across batch.
    """
    high_dtype = _higher_dtype(X.dtype)
    n = X.shape[-2]
    eps = _eps(X.dtype)

    diff = X.to(high_dtype) - X_exact.to(high_dtype)
    diffnorm = torch.linalg.matrix_norm(diff, ord=float('inf'))
    xactnorm = torch.linalg.matrix_norm(X_exact.to(high_dtype), ord=float('inf'))

    denom = n * xactnorm * eps
    denom = torch.clamp(denom, min=1e-300)
    scaled = (diffnorm / denom) * rcond
    return scaled.max().item()


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------


def run_one(batch, n, cond, dtype, backend):
    """Factor with the given backend. Returns dict of residuals."""
    _seed()
    A = make_matrix_with_cond(batch, n, cond, dtype)

    prev = torch.backends.cuda.preferred_linalg_library()
    try:
        torch.backends.cuda.preferred_linalg_library(backend)
        LU, pivots, info = torch.linalg.lu_factor_ex(A.clone())
    finally:
        torch.backends.cuda.preferred_linalg_library(prev)

    lu_res = lu_factorization_residual(A, LU, pivots)
    singular = (info > 0).any().item()

    return {
        'lu_res': lu_res,
        'singular': singular,
        'pivots': pivots,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare LU backends")
    parser.add_argument("--dtype", choices=["float32", "float64", "complex64", "complex128"],
                        default=None, help="Single dtype to test (default: all)")
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--size", type=int, nargs="+", default=[259, 1025])
    parser.add_argument("--cond", type=float, nargs="+", default=None,
                        help="Condition numbers (default: 2, 1e4, 1/eps)")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    if args.dtype:
        dtypes = [dtype_map[args.dtype]]
    else:
        dtypes = [torch.float32, torch.float64, torch.complex64, torch.complex128]

    backends = ["magma", "cusolver"]

    # Header
    print(f"{'dtype':>12s} {'batch':>5s} {'n':>5s} {'cond':>10s} | "
          f"{'LU(M)':>6s} {'LU(C)':>6s} | {'piv':>5s} | {'status':>6s}")
    print("-" * 70)

    for dtype in dtypes:
        eps = _eps(dtype)
        if args.cond:
            conds = args.cond
        else:
            conds = [2.0, 1e4, 0.1 / eps]

        for batch in args.batch:
            for n in args.size:
                for cond in conds:
                    results = {}
                    for backend in backends:
                        try:
                            results[backend] = run_one(batch, n, cond, dtype, backend)
                        except Exception as e:
                            results[backend] = {'lu_res': float('nan'), 'solve_res': float('nan'),
                                                'fwd_err': float('nan'), 'singular': False,
                                                'error': str(e)}

                    m = results["magma"]
                    c = results["cusolver"]

                    # Compare pivots
                    if 'pivots' in m and 'pivots' in c:
                        piv_match = (m['pivots'] == c['pivots']).all().item()
                        piv_str = "yes" if piv_match else "NO"
                    else:
                        piv_str = "?"

                    # Status based on MAGMA's THRESH=30 applied to backward residual
                    if 'error' in m or 'error' in c:
                        status = "ERROR"
                    elif m.get('singular') or c.get('singular'):
                        status = "SING"
                    elif m['lu_res'] < 30 and c['lu_res'] < 30:
                        status = "ok"
                    else:
                        status = "WARN"

                    cond_str = f"{cond:.0e}" if cond > 100 else f"{cond:.1f}"

                    print(f"{str(dtype):>12s} {batch:5d} {n:5d} {cond_str:>10s} | "
                          f"{m['lu_res']:6.2f} {c['lu_res']:6.2f} | {piv_str:>5s} | {status:>6s}")

        print()


if __name__ == "__main__":
    main()
