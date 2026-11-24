# Changelog

## 1.0.0 — defensibility pass

Rebuilt the project around claims that can be defended mathematically and statistically.

- Corrected the annualisation in the Gaussian posterior update. A weekly return annualised by multiplication with 52 has observation-noise variance `52 * w'Σw`; the previous implementation overstated Fisher information.
- Corrected the A-optimal generalised-eigenproblem to use `52*Σ + C` and replaced non-symmetric `eig(solve(B,A))` calculations with Cholesky-whitened symmetric eigendecompositions.
- Renamed the former “E-optimal” rule to a **max-uncertainty / E-inspired heuristic** because the top covariance eigenvector is not generally the exact one-step E-optimal solution under anisotropic noise.
- Reworked policy comparisons to use common simulated market shocks where appropriate.
- Rebuilt model discrimination as a coherent active-query problem: all cheap contexts are visible, but only three high-fidelity responses can be queried per round. Designed and random policies are evaluated on identical simulated worlds.
- Added an informativeness-heterogeneity sensitivity study. The baseline still supports the public “roughly one-third as many observations” result, while the sensitivity analysis shows clearly that the gain disappears when query opportunities are homogeneous.
- Reframed turnover as **probe reconfiguration distance** rather than presenting it as an inherent trading cost of learning.
- Rebuilt the exploit study around paired bootstrap uncertainty. A/D have higher point-estimate synthetic Sharpe than random in the baseline, but their 95% paired intervals include zero, so the README no longer claims superior investment performance.
- Added `docs/METHODOLOGY.md`, stronger tests, reproducible report generation and explicit synthetic-data / no-track-record limitations.

## 0.6.0 and earlier

Historical development versions explored alpha learning, model discrimination, reconfiguration/turnover proxies and a downstream exploit study. Version 1.0.0 supersedes their quantitative claims where the corrected measurement model changes the result.
