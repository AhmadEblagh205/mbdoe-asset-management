"""Active model discrimination under a limited high-fidelity query budget.

The setting is synthetic but deliberately coherent. Each period exposes a
cheap context ``x`` for every candidate asset. Two rival models predict a
costly high-fidelity response:

    M+ : y = +phi*x + eps
    M- : y = -phi*x + eps

Half the assets follow each model. The researcher may query only ``k`` of the
responses per period. The designed policy spends those queries where the
standardised rival predictions disagree most; the random policy spends the
same query budget uniformly.

The key stress variable is *opportunity heterogeneity*. Context magnitudes are
lognormally dispersed. When that dispersion is near zero, candidate queries
are similarly informative and active design has little to exploit. As the
dispersion grows, a few observations become much more informative than the
rest and targeted querying becomes more valuable. The sensitivity analysis
makes that dependence explicit instead of presenting one speed-up as a
universal law.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscriminationConfig:
    n_assets: int = 10
    n_periods: int = 180
    k_watch: int = 3
    phi: float = 0.25
    response_noise: float = 0.04
    context_magnitude: float = 0.04
    informativeness_log_sd: float = 1.50
    max_context_multiplier: float = 10.0


@dataclass(frozen=True)
class SimulatedWorld:
    truth: np.ndarray
    contexts: np.ndarray
    responses: np.ndarray


def simulate_world(seed: int, cfg: DiscriminationConfig) -> SimulatedWorld:
    rng = np.random.default_rng(seed)
    n, T = cfg.n_assets, cfg.n_periods
    truth = np.array([1] * (n // 2) + [-1] * (n - n // 2), dtype=int)

    # Cheap candidate contexts are visible before the high-fidelity query.
    # The -0.5*sigma^2 shift keeps the *mean* positive multiplier near one as
    # heterogeneity changes, while log-SD controls how unequal opportunities
    # are. Clipping prevents a single pathological draw from dominating a run.
    signs = rng.choice([-1.0, 1.0], size=(T, n))
    sigma = cfg.informativeness_log_sd
    multipliers = np.exp(sigma * rng.normal(size=(T, n)) - 0.5 * sigma**2)
    cap = float(cfg.max_context_multiplier)
    multipliers = np.clip(multipliers, 1.0 / cap, cap)
    contexts = cfg.context_magnitude * signs * multipliers

    mean = truth[None, :] * cfg.phi * contexts
    responses = mean + rng.normal(0.0, cfg.response_noise, size=(T, n))
    return SimulatedWorld(truth=truth, contexts=contexts, responses=responses)


def _posterior_correct_probability(log_odds: np.ndarray, truth: np.ndarray) -> float:
    p_plus = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -50, 50)))
    p_correct = np.where(truth == 1, p_plus, 1.0 - p_plus)
    return float(np.mean(p_correct))


def run_policy(
    world: SimulatedWorld,
    cfg: DiscriminationConfig,
    designed: bool,
    random_seed: int = 0,
) -> np.ndarray:
    """Return average posterior P(correct model) after each query round."""
    rng = np.random.default_rng(random_seed)
    log_odds = np.zeros(cfg.n_assets, dtype=float)
    trajectory = []

    for t in range(cfg.n_periods):
        x = world.contexts[t]
        y = world.responses[t]
        plus = cfg.phi * x
        minus = -cfg.phi * x

        if designed:
            disagreement = (plus - minus) ** 2 / cfg.response_noise**2
            # Tiny random jitter gives fair tie-breaking in the homogeneous
            # sensitivity case without changing any genuine score ordering.
            jitter = rng.uniform(0.0, 1e-12, cfg.n_assets)
            watch = np.argsort(disagreement + jitter)[-cfg.k_watch :]
        else:
            watch = rng.choice(cfg.n_assets, size=cfg.k_watch, replace=False)

        i = watch
        ll_plus = -((y[i] - plus[i]) ** 2) / (2.0 * cfg.response_noise**2)
        ll_minus = -((y[i] - minus[i]) ** 2) / (2.0 * cfg.response_noise**2)
        log_odds[i] += ll_plus - ll_minus
        trajectory.append(_posterior_correct_probability(log_odds, world.truth))

    return np.asarray(trajectory)


def first_crossing(traj: np.ndarray, level: float = 0.95) -> int | None:
    idx = np.flatnonzero(np.asarray(traj) >= level)
    return int(idx[0] + 1) if idx.size else None


def paired_study(
    reps: int = 200,
    cfg: DiscriminationConfig | None = None,
    seed: int = 0,
) -> dict:
    """Compare designed and random attention on identical simulated worlds."""
    cfg = cfg or DiscriminationConfig()
    designed = []
    random = []
    designed_cross = []
    random_cross = []

    for rep in range(reps):
        world = simulate_world(seed + rep, cfg)
        d = run_policy(world, cfg, designed=True, random_seed=10_000 + rep)
        r = run_policy(world, cfg, designed=False, random_seed=20_000 + rep)
        designed.append(d)
        random.append(r)
        designed_cross.append(first_crossing(d))
        random_cross.append(first_crossing(r))

    d_arr = np.asarray(designed)
    r_arr = np.asarray(random)
    d_mean = d_arr.mean(axis=0)
    r_mean = r_arr.mean(axis=0)
    d_cross_mean = first_crossing(d_mean)
    r_cross_mean = first_crossing(r_mean)

    def crossing_summary(values: list[int | None]) -> dict:
        observed = np.asarray([v for v in values if v is not None], dtype=float)
        return {
            "crossed_fraction": float(len(observed) / len(values)),
            "median_periods_if_crossed": float(np.median(observed)) if len(observed) else None,
            "p10_periods_if_crossed": float(np.quantile(observed, 0.10)) if len(observed) else None,
            "p90_periods_if_crossed": float(np.quantile(observed, 0.90)) if len(observed) else None,
        }

    return {
        "designed_mean_trajectory": d_mean,
        "random_mean_trajectory": r_mean,
        "designed_periods_to_95_mean_trajectory": d_cross_mean,
        "random_periods_to_95_mean_trajectory": r_cross_mean,
        "designed_queries_to_95_mean_trajectory": None if d_cross_mean is None else d_cross_mean * cfg.k_watch,
        "random_queries_to_95_mean_trajectory": None if r_cross_mean is None else r_cross_mean * cfg.k_watch,
        "designed_crossing_distribution": crossing_summary(designed_cross),
        "random_crossing_distribution": crossing_summary(random_cross),
        "config": cfg,
    }


def heterogeneity_sensitivity(
    log_sds=(0.0, 0.5, 1.0, 1.5),
    reps: int = 80,
    n_periods: int = 240,
    level: float = 0.95,
) -> list[dict]:
    """Show how sample efficiency depends on query-opportunity heterogeneity."""
    rows = []
    for j, log_sd in enumerate(log_sds):
        cfg = DiscriminationConfig(
            n_periods=n_periods, informativeness_log_sd=float(log_sd)
        )
        d_curves, r_curves = [], []
        for rep in range(reps):
            world = simulate_world(50_000 + 1_000 * j + rep, cfg)
            d_curves.append(run_policy(world, cfg, True, 70_000 + rep))
            r_curves.append(run_policy(world, cfg, False, 80_000 + rep))
        d_cross = first_crossing(np.mean(d_curves, axis=0), level)
        r_cross = first_crossing(np.mean(r_curves, axis=0), level)
        speedup = None
        if d_cross is not None and r_cross is not None:
            speedup = float(r_cross / d_cross)
        rows.append({
            "informativeness_log_sd": float(log_sd),
            "designed_periods": d_cross,
            "random_periods": r_cross,
            "random_over_designed_speedup": speedup,
        })
    return rows


# Backwards-compatible helper used by the original API/tests.
def run(n_periods: int = 180, k_watch: int = 3, designed: bool = True, seed: int = 0):
    cfg = DiscriminationConfig(n_periods=n_periods, k_watch=k_watch)
    world = simulate_world(seed, cfg)
    return run_policy(world, cfg, designed=designed, random_seed=seed + 100_000)
