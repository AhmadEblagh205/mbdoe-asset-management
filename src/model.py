"""Linear-Gaussian sequential measurement model used in study 1.

This is deliberately a *synthetic measurement-design analogy*, not a claim
that public-market returns are unobservable unless a portfolio is held.

The latent annualised alpha vector is ``alpha``.  A one-period synthetic
probe with weights ``w`` produces an annualised scalar measurement

    z_t = w_t' alpha + eta_t
    eta_t ~ N(0, ppy * w_t' Sigma w_t)

where ``Sigma`` is the annualised asset-return covariance matrix and ``ppy``
is the number of observation periods per year.  This is exactly what results
if a weekly portfolio return is annualised by multiplication with ``ppy``.

With a Gaussian prior, the posterior remains Gaussian.  The one-step Fisher
information contribution is

    w w' / (ppy * w' Sigma w)

so *direction* determines information; the scaling of ``w`` is used only to
put all candidate probes on the same risk/gross-exposure convention.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Universe:
    """Synthetic latent-alpha world shared by the counterfactual policies."""

    n_assets: int = 8
    seed: int = 0
    alpha_scale: float = 0.04
    vol_range: tuple = (0.12, 0.30)
    avg_corr: float = 0.35
    periods_per_year: int = 52

    rng: np.random.Generator = field(init=False)
    alpha: np.ndarray = field(init=False)
    Sigma: np.ndarray = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        n = self.n_assets
        self.alpha = self.rng.normal(0.0, self.alpha_scale, n)
        vols = self.rng.uniform(*self.vol_range, n)
        corr = np.full((n, n), self.avg_corr)
        corr += (1.0 - self.avg_corr) * np.eye(n)
        self.Sigma = np.outer(vols, vols) * corr

    def draw_annualised_asset_return(self) -> np.ndarray:
        """One synthetic period, expressed in annualised return units.

        A weekly innovation has covariance ``Sigma / ppy``.  Multiplying the
        weekly return by ``ppy`` gives an annualised observation with noise
        covariance ``ppy * Sigma``.
        """
        ppy = self.periods_per_year
        eps_weekly = self.rng.multivariate_normal(
            np.zeros(self.n_assets), self.Sigma / ppy
        )
        return self.alpha + ppy * eps_weekly

    def observe(self, w: np.ndarray, annualised_return: np.ndarray | None = None) -> float:
        """Return the scalar annualised measurement for probe ``w``.

        Passing ``annualised_return`` lets several counterfactual policies be
        scored against the same simulated market shock (common random numbers).
        """
        r = self.draw_annualised_asset_return() if annualised_return is None else annualised_return
        return float(w @ r)


class BayesAlpha:
    """Conjugate Gaussian posterior for the annualised alpha vector."""

    def __init__(
        self,
        n: int,
        Sigma: np.ndarray,
        prior_sd: float,
        periods_per_year: int,
    ):
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.ppy = int(periods_per_year)
        self.P = np.eye(n) / float(prior_sd) ** 2
        self.b = np.zeros(n, dtype=float)

    def measurement_variance(self, w: np.ndarray) -> float:
        return self.ppy * float(w @ self.Sigma @ w)

    def update(self, w: np.ndarray, z: float) -> None:
        """Bayesian update for ``z = w'alpha + N(0, ppy*w'Sigma*w)``."""
        noise_var = self.measurement_variance(w)
        if not np.isfinite(noise_var) or noise_var <= 0:
            raise ValueError("probe must have positive finite measurement variance")
        self.P += np.outer(w, w) / noise_var
        self.b += w * float(z) / noise_var

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.P)

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.P, self.b)

    def trace_variance(self) -> float:
        """A-optimal objective: total marginal posterior variance."""
        return float(np.trace(self.cov))

    def logdet_precision(self) -> float:
        """D-optimal objective: log determinant of posterior precision."""
        sign, val = np.linalg.slogdet(self.P)
        if sign <= 0:
            raise ValueError("posterior precision is not positive definite")
        return float(val)

    def ci_half_width(self) -> np.ndarray:
        """Per-asset 95% marginal credible-interval half-widths."""
        return 1.96 * np.sqrt(np.diag(self.cov))
