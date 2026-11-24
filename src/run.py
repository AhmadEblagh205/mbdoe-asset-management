"""Run the sequential-design studies and regenerate all report artefacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from design import (
    design_probe_a,
    design_probe_d,
    design_probe_max_uncertainty,
    random_probe,
    static_probe,
)
from discriminate import DiscriminationConfig, heterogeneity_sensitivity, paired_study
from model import BayesAlpha, Universe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def alpha_study(periods: int, reps: int, prior_sd: float = 0.20, seed: int = 0):
    names = [
        "A-optimal",
        "D-optimal",
        "max-uncertainty",
        "random",
        "static",
    ]
    trace_curves = {k: [] for k in names}
    logdet_curves = {k: [] for k in names}
    reconfiguration_curves = {k: [] for k in names}

    for rep in range(reps):
        uni = Universe(seed=seed + rep)
        random_rng = np.random.default_rng(10_000 + rep)
        posts = {
            k: BayesAlpha(uni.n_assets, uni.Sigma, prior_sd, uni.periods_per_year)
            for k in names
        }
        history_trace = {k: [] for k in names}
        history_logdet = {k: [] for k in names}
        history_turn = {k: [] for k in names}
        previous = {k: np.zeros(uni.n_assets) for k in names}
        w_static = static_probe(uni.n_assets, uni.Sigma)

        for _ in range(periods):
            common_return = uni.draw_annualised_asset_return()
            probes = {
                "A-optimal": design_probe_a(posts["A-optimal"], uni.Sigma),
                "D-optimal": design_probe_d(posts["D-optimal"], uni.Sigma),
                "max-uncertainty": design_probe_max_uncertainty(posts["max-uncertainty"], uni.Sigma),
                "random": random_probe(random_rng, uni.n_assets, uni.Sigma),
                "static": w_static,
            }
            for name, w in probes.items():
                posts[name].update(w, uni.observe(w, common_return))
                history_trace[name].append(posts[name].trace_variance())
                history_logdet[name].append(posts[name].logdet_precision())
                history_turn[name].append(float(np.abs(w - previous[name]).sum()))
                previous[name] = w

        for name in names:
            trace_curves[name].append(history_trace[name])
            logdet_curves[name].append(history_logdet[name])
            reconfiguration_curves[name].append(np.cumsum(history_turn[name]))

    return (
        {k: np.mean(v, axis=0) for k, v in trace_curves.items()},
        {k: np.mean(v, axis=0) for k, v in logdet_curves.items()},
        {k: np.mean(v, axis=0) for k, v in reconfiguration_curves.items()},
    )


def _first_at_or_below(curve: np.ndarray, target: float) -> int | None:
    idx = np.flatnonzero(np.asarray(curve) <= target)
    return int(idx[0] + 1) if idx.size else None


def _plot_alpha(trace_curves, logdet_curves, periods: int) -> None:
    weeks = np.arange(1, periods + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for name, curve in trace_curves.items():
        axes[0].plot(weeks, curve, lw=1.8, label=name)
    axes[0].set(
        xlabel="Weekly query rounds",
        ylabel="trace posterior covariance",
        title="A-optimal objective: total uncertainty",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    for name, curve in logdet_curves.items():
        axes[1].plot(weeks, curve, lw=1.8, label=name)
    axes[1].set(
        xlabel="Weekly query rounds",
        ylabel="log det posterior precision",
        title="D-optimal objective: information volume",
    )
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "alpha_learning.png", dpi=170)
    plt.close(fig)


def _plot_reconfiguration(trace_curves, reconfiguration_curves) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for name in trace_curves:
        ax.plot(reconfiguration_curves[name], trace_curves[name], lw=1.8, label=name)
    ax.set(
        xlabel="Cumulative L1 probe reconfiguration",
        ylabel="trace posterior covariance",
        title="Information versus probe reconfiguration",
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_per_cost.png", dpi=170)
    plt.close(fig)


def _plot_discrimination(discrimination: dict) -> None:
    d = discrimination["designed_mean_trajectory"]
    r = discrimination["random_mean_trajectory"]
    periods = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.plot(periods, d, lw=1.9, label="designed high-fidelity queries")
    ax.plot(periods, r, lw=1.9, label="random queries")
    ax.axhline(0.95, ls="--", lw=1.0)
    ax.set(
        xlabel="Query rounds (3 high-fidelity observations per round)",
        ylabel="Mean posterior probability of the correct model",
        title="Sequential model discrimination",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_discrimination.png", dpi=170)
    plt.close(fig)


def _plot_sensitivity(rows: list[dict]) -> None:
    x = [row["informativeness_log_sd"] for row in rows]
    y = [row["random_over_designed_speedup"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(x, y, marker="o", lw=1.8)
    ax.axhline(1.0, ls="--", lw=1.0)
    ax.set(
        xlabel="Log-SD of query informativeness",
        ylabel="Random / designed rounds to 95% confidence",
        title="Active design pays more when opportunities are heterogeneous",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "discrimination_sensitivity.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=156)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--prior-sd", type=float, default=0.20)
    ap.add_argument("--disc-reps", type=int, default=200)
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    trace_curves, logdet_curves, reconfig = alpha_study(
        args.periods, args.reps, args.prior_sd
    )
    _plot_alpha(trace_curves, logdet_curves, args.periods)
    _plot_reconfiguration(trace_curves, reconfig)

    random_final = float(trace_curves["random"][-1])
    alpha_summary = {
        "prior_sd": args.prior_sd,
        "final_trace_variance": {k: round(float(v[-1]), 6) for k, v in trace_curves.items()},
        "final_logdet_precision": {k: round(float(v[-1]), 4) for k, v in logdet_curves.items()},
        "rounds_to_random_final_trace": {
            k: _first_at_or_below(v, random_final) for k, v in trace_curves.items()
        },
        "final_probe_reconfiguration_l1": {
            k: round(float(v[-1]), 2) for k, v in reconfig.items()
        },
    }

    cfg = DiscriminationConfig()
    discrimination = paired_study(reps=args.disc_reps, cfg=cfg)
    _plot_discrimination(discrimination)

    sensitivity = heterogeneity_sensitivity(
        log_sds=(0.0, 0.5, 1.0, 1.5), reps=80, n_periods=240, level=0.95
    )
    _plot_sensitivity(sensitivity)

    disc_summary = {
        "config": {
            "n_assets": cfg.n_assets,
            "n_periods": cfg.n_periods,
            "k_watch": cfg.k_watch,
            "informativeness_log_sd": cfg.informativeness_log_sd,
            "max_context_multiplier": cfg.max_context_multiplier,
        },
        "periods_to_95_mean_trajectory": {
            "designed": discrimination["designed_periods_to_95_mean_trajectory"],
            "random": discrimination["random_periods_to_95_mean_trajectory"],
        },
        "queries_to_95_mean_trajectory": {
            "designed": discrimination["designed_queries_to_95_mean_trajectory"],
            "random": discrimination["random_queries_to_95_mean_trajectory"],
        },
        "crossing_distribution": {
            "designed": discrimination["designed_crossing_distribution"],
            "random": discrimination["random_crossing_distribution"],
        },
        "heterogeneity_sensitivity": sensitivity,
    }

    summary = {"alpha_learning": alpha_summary, "model_discrimination": disc_summary}
    (ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
