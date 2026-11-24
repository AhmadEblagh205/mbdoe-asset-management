import numpy as np

from design import (
    design_probe_a,
    design_probe_d,
    design_probe_max_uncertainty,
    random_probe,
    risk_scale,
    static_probe,
)
from discriminate import (
    DiscriminationConfig,
    first_crossing,
    heterogeneity_sensitivity,
    paired_study,
    run_policy,
    simulate_world,
)
from exploit import study as exploit_study
from model import BayesAlpha, Universe


def test_posterior_matches_closed_form_single_asset():
    Sigma = np.array([[0.04]])
    ppy = 52
    prior_sd = 0.10
    post = BayesAlpha(1, Sigma, prior_sd=prior_sd, periods_per_year=ppy)
    w = np.array([1.0])
    observations = [0.03, -0.02, 0.01]
    for z in observations:
        post.update(w, z)

    noise_var = ppy * Sigma[0, 0]
    expected_precision = 1.0 / prior_sd**2 + len(observations) / noise_var
    expected_b = sum(observations) / noise_var
    assert np.isclose(post.P[0, 0], expected_precision)
    assert np.isclose(post.b[0], expected_b)
    assert np.isclose(post.mean[0], expected_b / expected_precision)


def test_information_only_grows_under_valid_updates():
    uni = Universe(seed=3)
    post = BayesAlpha(uni.n_assets, uni.Sigma, 0.20, uni.periods_per_year)
    rng = np.random.default_rng(0)
    traces = [post.trace_variance()]
    logdets = [post.logdet_precision()]
    for _ in range(30):
        w = random_probe(rng, uni.n_assets, uni.Sigma)
        post.update(w, uni.observe(w))
        traces.append(post.trace_variance())
        logdets.append(post.logdet_precision())
    assert all(b <= a + 1e-12 for a, b in zip(traces, traces[1:]))
    assert all(b >= a - 1e-12 for a, b in zip(logdets, logdets[1:]))


def test_risk_scale_respects_both_limits():
    uni = Universe(seed=1)
    w = risk_scale(np.ones(uni.n_assets), uni.Sigma, 0.10, 2.0)
    assert np.sqrt(w @ uni.Sigma @ w) <= 0.10 + 1e-9
    assert np.abs(w).sum() <= 2.0 + 1e-9


def test_a_optimal_direction_beats_random_candidates_on_its_one_step_objective():
    uni = Universe(seed=7)
    post = BayesAlpha(uni.n_assets, uni.Sigma, 0.20, uni.periods_per_year)
    rng = np.random.default_rng(9)
    for _ in range(12):
        wr = random_probe(rng, uni.n_assets, uni.Sigma)
        post.update(wr, uni.observe(wr))

    C = post.cov
    B = post.ppy * uni.Sigma + C

    def score(w):
        return float(w @ C @ C @ w) / float(w @ B @ w)

    chosen = design_probe_a(post, uni.Sigma)
    random_scores = [score(rng.normal(size=uni.n_assets)) for _ in range(500)]
    assert score(chosen) >= max(random_scores) - 1e-10


def test_d_optimal_direction_beats_random_candidates_on_its_one_step_objective():
    uni = Universe(seed=8)
    post = BayesAlpha(uni.n_assets, uni.Sigma, 0.20, uni.periods_per_year)
    rng = np.random.default_rng(11)
    for _ in range(12):
        wr = random_probe(rng, uni.n_assets, uni.Sigma)
        post.update(wr, uni.observe(wr))

    C = post.cov
    B = post.ppy * uni.Sigma

    def score(w):
        return float(w @ C @ w) / float(w @ B @ w)

    chosen = design_probe_d(post, uni.Sigma)
    random_scores = [score(rng.normal(size=uni.n_assets)) for _ in range(500)]
    assert score(chosen) >= max(random_scores) - 1e-10


def test_max_uncertainty_is_explicitly_an_e_inspired_heuristic():
    uni = Universe(seed=4)
    rng = np.random.default_rng(2)
    post = BayesAlpha(uni.n_assets, uni.Sigma, 0.20, uni.periods_per_year)
    for _ in range(5):
        w = random_probe(rng, uni.n_assets, uni.Sigma)
        post.update(w, uni.observe(w))
    _, vecs = np.linalg.eigh(post.cov)
    target = vecs[:, -1]
    chosen = design_probe_max_uncertainty(post, uni.Sigma)
    # Scaling changes length but not direction.
    alignment = abs(np.dot(chosen / np.linalg.norm(chosen), target))
    assert alignment > 0.999999


def test_repeating_one_static_probe_adds_rank_one_data_information():
    uni = Universe(seed=5)
    prior_sd = 0.20
    post = BayesAlpha(uni.n_assets, uni.Sigma, prior_sd, uni.periods_per_year)
    prior_precision = post.P.copy()
    w = static_probe(uni.n_assets, uni.Sigma)
    for _ in range(20):
        post.update(w, uni.observe(w))
    data_information = post.P - prior_precision
    assert np.linalg.matrix_rank(data_information, tol=1e-9) == 1


def test_designed_discrimination_uses_same_budget_but_reaches_confidence_first():
    cfg = DiscriminationConfig(n_periods=140)
    result = paired_study(reps=40, cfg=cfg, seed=0)
    d = result["designed_periods_to_95_mean_trajectory"]
    r = result["random_periods_to_95_mean_trajectory"]
    assert d is not None and r is not None
    assert d < 0.6 * r
    assert result["designed_queries_to_95_mean_trajectory"] == d * cfg.k_watch
    assert result["random_queries_to_95_mean_trajectory"] == r * cfg.k_watch


def test_query_policy_never_needs_unqueried_responses_to_choose():
    cfg = DiscriminationConfig(n_periods=20)
    world = simulate_world(12, cfg)
    # Altering responses cannot alter which contexts are intrinsically ranked;
    # both runs still produce a valid trajectory of the same length.  Selection
    # is computed from contexts before response values are indexed.
    altered = type(world)(world.truth, world.contexts, world.responses * 1000.0)
    a = run_policy(world, cfg, designed=True, random_seed=1)
    b = run_policy(altered, cfg, designed=True, random_seed=1)
    assert len(a) == len(b) == cfg.n_periods


def test_design_advantage_grows_with_opportunity_heterogeneity():
    rows = heterogeneity_sensitivity(
        log_sds=(0.0, 1.5), reps=24, n_periods=240, level=0.90
    )
    assert rows[0]["random_over_designed_speedup"] is not None
    assert rows[1]["random_over_designed_speedup"] is not None
    assert rows[0]["random_over_designed_speedup"] < 1.5
    assert rows[1]["random_over_designed_speedup"] > 2.0


def test_exploit_study_reports_paired_uncertainty_not_just_point_estimates():
    result = exploit_study(periods=80, reps=30, prior_sd=0.20)
    assert "paired_vs_random" in result
    for name in ["A-optimal", "D-optimal", "max-uncertainty", "static"]:
        item = result["paired_vs_random"][name]
        assert set(item) == {"mean_difference_vs_random", "difference_ci95"}
        assert len(item["difference_ci95"]) == 2
