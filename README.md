# Sequential Model Identification with MBDoE — simulated market research

[![CI](https://github.com/AhmadEblagh205/mbdoe-asset-management/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/mbdoe-asset-management/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this to carry ideas from my PhD on model-based design of experiments (MBDoE) into a financial research setting. When measurements are limited, the next one should be chosen for the information it gives. Everything here is simulated, and it is a methods project, not a trading strategy or a track record.

## What it does

- Study 1 learns a hidden eight-dimensional alpha vector from one linear measurement per weekly round, using A-optimal, D-optimal, max-uncertainty, random and static measurement policies.
- Study 2 identifies which of two rival response models governs ten synthetic candidates. Cheap context is visible for all of them, but only three high-fidelity observations are allowed per round.
- A sensitivity run varies how unequal the candidates are in informativeness.
- An exploit study scores each policy's posterior on the synthetic true alpha.

The "probe portfolio" in study 1 is a linear-Gaussian measurement vector, used to show the mathematics of sequential design. Study 2 is the closer analogy to research allocation.

## Results

All results are from simulated data.

Across 200 paired simulated worlds, the designed policy reaches 95% posterior probability on the correct model in 33 rounds (99 high-fidelity observations). Random querying needs 104 rounds (312 observations), roughly three times as many.

The gain depends on how much candidates differ in informativeness. Random querying needs this many times as many rounds as the designed policy:

| Log-SD of informativeness | Random / designed |
|---:|---:|
| 0.0 | 1.02 |
| 0.5 | 2.06 |
| 1.0 | 3.13 |
| 1.5 | 3.19 |

In study 1, after 156 weekly rounds, A-optimal reaches the random policy's final total posterior variance after 101 rounds and D-optimal after 112.

For investment value the picture is weaker. After 208 probe rounds over 200 paired worlds, mean synthetic Sharpe is:

- A-optimal 0.163, D-optimal 0.177;
- random 0.131, static 0.020;
- an oracle that knows the true alpha 0.680.

The paired 95% bootstrap intervals for the gain over random are -0.015 to 0.082 (A) and -0.003 to 0.096 (D). Both include zero, so the downstream gain is inconclusive.

Figures are in `reports/figures/` and the numbers are in `reports/summary.json` and `reports/exploit_summary.json`.

## How to run

```bash
pip install -r requirements.txt
python src/run.py --periods 156 --reps 60 --disc-reps 200
python src/exploit.py --periods 208 --reps 200 --prior-sd 0.20
pytest -q
```

The tests check:

- the closed-form Bayesian update and monotonic information growth;
- the risk and gross constraints;
- the A- and D-optimal one-step objectives;
- rank-one information from a repeated static probe;
- active discrimination under an equal budget;
- the heterogeneity sensitivity and the paired downstream intervals.

CI runs them on push and pull request.

## Notes

The max-uncertainty policy is E-inspired and not exact E-optimal, because anisotropic observation noise matters. The probe reconfiguration plot shows cumulative L1 change in probe weights, which is not a trading cost. Equations and assumptions are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Licence

MIT, see [`LICENSE`](LICENSE).
