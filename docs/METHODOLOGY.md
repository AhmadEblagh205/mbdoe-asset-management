# Methodology notes

This document records the mathematical assumptions behind the simulations so that the headline results can be defended without pretending the setup is a live trading system.

## 1. Linear-Gaussian alpha learning

The latent parameter is an annualised alpha vector

\[
\alpha \in \mathbb{R}^n.
\]

A synthetic probe direction \(w_t\) produces one annualised scalar measurement

\[
z_t = w_t^\top \alpha + \eta_t,
\qquad
\eta_t \sim \mathcal{N}\!\left(0,\;m\,w_t^\top\Sigma w_t\right),
\]

where \(m=52\) weekly periods per year. This is the distribution obtained by multiplying a weekly portfolio return by 52. The probe is a **measurement vector in a stylised linear experiment**. Public-market returns are of course observable without holding a portfolio; the simulation is an experiment-design analogy, not a claim about market-data access.

With a Gaussian prior \(\alpha\sim\mathcal{N}(0,C_0)\), posterior precision updates as

\[
P_{t+1}=P_t+\frac{w_tw_t^\top}{m\,w_t^\top\Sigma w_t}.
\]

The repository uses a 20% prior standard deviation per alpha coordinate. The true synthetic alpha dispersion is 4%.

### A-optimal query

For posterior covariance \(C=P^{-1}\), Sherman-Morrison gives the reduction in \(\operatorname{tr}(C)\) as

\[
\frac{w^\top C^2 w}{m\,w^\top\Sigma w+w^\top Cw}.
\]

The maximizing direction is the leading generalised eigenvector of \(C^2\) against \(m\Sigma+C\).

### D-optimal query

The matrix determinant lemma gives a determinant gain proportional to

\[
1+\frac{w^\top Cw}{m\,w^\top\Sigma w}.
\]

The maximizing direction is the leading generalised eigenvector of \(C\) against \(m\Sigma\).

### Max-uncertainty heuristic

The third designed policy probes the leading eigenvector of \(C\), the direction with the largest absolute posterior variance. It is **E-inspired**, but is not labelled exact E-optimality because anisotropic observation noise means that the one-step E-optimal solution need not coincide with the largest-eigenvalue direction of \(C\).

## 2. Model discrimination

Each synthetic candidate has a cheap context \(x\). Two rival models predict a costly high-fidelity response:

\[
M_+: y=+\phi x+\epsilon, \qquad
M_-: y=-\phi x+\epsilon,
\]

with Gaussian response noise. Half the candidates follow each model. Only three responses may be queried per round.

The designed policy ranks candidate queries by standardized squared disagreement

\[
\frac{(\mu_+-\mu_-)^2}{\sigma^2}
\]

and queries the three largest values. Both designed and random policies are scored on the **same pre-generated simulated worlds**, so the query rule is the only systematic difference.

### Why the baseline has heterogeneous query opportunities

Context magnitudes have lognormal dispersion. This is a synthetic way to create research opportunities with unequal expected informativeness. The sensitivity analysis varies that dispersion. At zero heterogeneity the designed and random policies need almost the same number of rounds; as heterogeneity rises, active design becomes materially more sample-efficient. Therefore the roughly one-third headline is a property of the stated synthetic baseline, not a universal MBDoE constant.

## 3. Downstream decision-value stress test

After the alpha-learning phase, each policy forms the same constrained mean-variance direction from its posterior mean alpha. The simulator then evaluates that direction using the latent true alpha.

The resulting quantity is labelled **synthetic true-alpha Sharpe**. It is not a historical backtest, live return, track record or investable performance claim. Policy comparisons are paired by simulated world, and the report includes bootstrap confidence intervals for the mean difference versus random probing.

## 4. Reconfiguration metric

`learning_per_cost.png` uses cumulative L1 change in probe weights. It is best interpreted as **probe reconfiguration distance**. If the synthetic measurement vectors were implemented as tradable portfolios, multiplying that distance by a transaction-cost assumption would produce an illustrative cost, but the repository no longer presents that conversion as an inherent cost of learning.


## Notes moved from the README

- Study 1 hides an eight-dimensional annualised alpha vector. Each weekly round gives one scalar measurement z_t = w_t'alpha + eta_t with eta_t ~ N(0, 52 w_t' Sigma w_t), and the posterior update is conjugate Gaussian. The five policies are A-optimal (greedy minimum of trace(C)), D-optimal (greedy maximum of log det(P)), the max-uncertainty heuristic, random probes and a static direction.
- A and D directions are solved as symmetric generalised eigenproblems. The max-uncertainty policy is E-inspired, not exact E-optimal.
- After 156 weekly rounds, A-optimal reaches the random policy's final total posterior variance after 101 rounds and D-optimal after 112. A repeated static direction adds only rank-one information, so it barely explores the parameter space.
- Exploit study, 208 probe rounds and 200 paired worlds, mean synthetic Sharpe: A-optimal 0.163, D-optimal 0.177, random 0.131, static 0.020, oracle with the true alpha 0.680. The paired 95% bootstrap interval for the improvement over random is -0.015 to 0.082 (A) and -0.003 to 0.096 (D), so both include zero. The max-uncertainty heuristic does worse than random in this study.
- Sensitivity of discrimination to the log-SD of informativeness (designed rounds, random rounds): 0.0 gives 193 and 197; 0.5 gives 68 and 140; 1.0 gives 38 and 119; 1.5 gives 32 and 102.
