"""Sequential probe-design rules for the synthetic alpha-learning study."""
import numpy as np

from model import BayesAlpha


def risk_scale(
    w: np.ndarray,
    Sigma: np.ndarray,
    risk_budget: float,
    gross_cap: float,
) -> np.ndarray:
    """Scale a direction to a common annual-volatility budget and gross cap."""
    w = np.asarray(w, dtype=float)
    vol2 = float(w @ Sigma @ w)
    if not np.isfinite(vol2) or vol2 <= 0:
        raise ValueError("probe direction must have positive finite variance")
    w = w * (float(risk_budget) / np.sqrt(vol2))
    gross = float(np.abs(w).sum())
    if gross > gross_cap:
        w = w * (float(gross_cap) / gross)
    return w


def _top_generalised_eigenvector(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return argmax_v v'Av / v'Bv for symmetric A and SPD B.

    Cholesky whitening keeps the numerical problem symmetric, avoiding the
    non-symmetric ``eig(solve(B, A))`` construction.
    """
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    L = np.linalg.cholesky(B)
    Linv = np.linalg.inv(L)
    M = Linv @ A @ Linv.T
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    q = vecs[:, int(np.argmax(vals))]
    return np.linalg.solve(L.T, q)


def design_probe_a(
    post: BayesAlpha,
    Sigma: np.ndarray,
    risk_budget: float = 0.10,
    gross_cap: float = 2.0,
) -> np.ndarray:
    """Greedy A-optimal probe.

    For posterior covariance C and annualised measurement noise
    ``s2 = ppy * w'Sigma w``, Sherman-Morrison gives a one-step reduction in
    trace(C) proportional to

        w'C^2w / (ppy*w'Sigma*w + w'Cw).

    The best direction is therefore the leading generalised eigenvector of
    ``C^2`` against ``ppy*Sigma + C``.
    """
    C = post.cov
    direction = _top_generalised_eigenvector(C @ C, post.ppy * Sigma + C)
    return risk_scale(direction, Sigma, risk_budget, gross_cap)


def design_probe_d(
    post: BayesAlpha,
    Sigma: np.ndarray,
    risk_budget: float = 0.10,
    gross_cap: float = 2.0,
) -> np.ndarray:
    """Greedy D-optimal probe.

    The determinant lemma shows that the one-step determinant gain depends on
    ``w'Cw / (ppy*w'Sigma*w)``.  Its maximiser is the leading generalised
    eigenvector of ``C`` against ``ppy*Sigma``.
    """
    C = post.cov
    direction = _top_generalised_eigenvector(C, post.ppy * Sigma)
    return risk_scale(direction, Sigma, risk_budget, gross_cap)


def design_probe_max_uncertainty(
    post: BayesAlpha,
    Sigma: np.ndarray,
    risk_budget: float = 0.10,
    gross_cap: float = 2.0,
) -> np.ndarray:
    """E-inspired heuristic: probe the largest posterior-variance direction.

    This is intentionally *not* called exact E-optimality.  With anisotropic
    observation noise, the exact one-step E-optimal problem is not generally
    solved by the top eigenvector of C.  The heuristic is kept as an explicit
    comparison because it illustrates the cost of ignoring the noise geometry.
    """
    vals, vecs = np.linalg.eigh(post.cov)
    direction = vecs[:, int(np.argmax(vals))]
    return risk_scale(direction, Sigma, risk_budget, gross_cap)


def random_probe(
    rng: np.random.Generator,
    n: int,
    Sigma: np.ndarray,
    risk_budget: float = 0.10,
    gross_cap: float = 2.0,
) -> np.ndarray:
    return risk_scale(rng.normal(size=n), Sigma, risk_budget, gross_cap)


def static_probe(
    n: int,
    Sigma: np.ndarray,
    risk_budget: float = 0.10,
    gross_cap: float = 2.0,
) -> np.ndarray:
    """Fixed long-only direction used as a deliberately non-exploratory baseline."""
    return risk_scale(np.ones(n) / n, Sigma, risk_budget, gross_cap)


# Backwards-compatible alias for the original public function name.
design_probe = design_probe_a
market_probe = static_probe
