# HMM-Diffusion

Regime-conditional synthetic financial time series. Volatility regimes are detected on S&P 500 total
return log returns, one unconditional diffusion specialist is trained per regime, and a hidden Markov
model estimates the regime path used to stitch the specialists' output into a single series.

This reproduces the pipeline of `Supervised HMMs.ipynb` from the HMM-GAN project with the GAN replaced
by a DDPM with a UniTST_MP transformer backbone. There is no TensorFlow anywhere in this repository.

## Objective

The HMM is not the generator. Each specialist learns the return distribution of one volatility regime
in isolation; the HMM supplies a regime path, and each step of that path selects which specialist's
pool the next value is drawn from. Two things are measured at once: how well the specialists reproduce
their regime's distribution, and how well the HMM recovers the regime path from emissions alone.

## Layout

```
configs/default.yaml      every constant in the pipeline; scripts and notebook both read it
scripts/                  stages 1-4, run in order
src/hmmdiff/              data, regimes, windows, pools, stitching, HMM wrappers, plots
third_party/diffusion/    vendored DDPM training and sampling code
third_party/hmmgan/       vendored TensorFlow-free HMM, state estimation, regime detection
HMM-Diffusion.ipynb       the analysis
```

`third_party/VENDORED.md` records where the vendored code came from and the one patch applied to it.

## Setup

```
pip install -r requirements.txt
```

On CUDA, install torch from the matching index first:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` uses loose bounds so the same file works on CUDA and Apple Silicon.
`requirements-ruya-freeze.txt` pins the exact CUDA 12.4 versions the diffusion code was tuned on; fall
back to it if resolution misbehaves.

## Running

```
python scripts/01_label_regimes.py            # A001 preprocess + regime labels (HMM input)
python scripts/02_build_diffusion_dataset.py  # attach 9 other assets; cut (N,128,10) windows
python scripts/03_train_specialists.py             # train 5 ten-channel specialists; needs a GPU
python scripts/04_generate_pools.py           # sample a pool from each specialist
jupyter lab HMM-Diffusion.ipynb
```

Stages 1 and 2 take under a minute on a laptop. Stage 3 is the expensive one. Stage 4 takes a few
minutes.

The notebook runs without stages 3 and 4 by substituting placeholder pools resampled from the real
per-regime returns, so plotting and the HMM comparison can be developed before training finishes. It
prints a warning when it does this. Placeholder results describe the resampler, not the diffusion
models.

Environment overrides for stage 3:

```
PYTHON=/path/to/python DEVICE=cuda EPOCHS=50 ./scripts/03_train_specialists.sh
REGIMES=0 EPOCHS=2 ./scripts/03_train_specialists.sh   # smoke test one regime
```



## Method

Two tracks share regime labels but differ in assets and scaling.

**HMM track (A001 only).** S&P 500 total return closes (`A001`) from 1988-01-19, log returns z-scored
using the mean and standard deviation of the whole series. The 9031 returns split 75/25 into 6773
train and 2258 test; the train portion splits 75/25 again into 5079 inner-training and 1694 validation
days. Regimes come from `Vol_Regime` on this z-scored A001 series. All four HMM variants and the
reference-style plots use A001 emissions only.

**Diffusion track (ten assets).** Stage 2 attaches the ruya ten-asset panel
(`A001, A004, A006, A008, A009, A011, A012, A013, A014, A015`) to the cached A001 regime labels by
date. Values are raw log returns; `Dataset_RegimeWindows` applies quantile scaling during training.
Specialists are unconditional DDPMs with `enc_in=10` and the full ruya loss
(`1.0-KL2_N+1.0-Corr+1.0-FFT`), which lets the model learn cross-asset correlations.

Because several assets start later than A001, the usable train overlap is 3395 days from 2001-01-01 to
2014-01-03 (the end of the outer train split), not the full 6773-day HMM training span. Regime labels
on that overlap are the stage-1 A001 labels mapped by date, not recomputed.

Sampling produces `n_pool` windows per regime, shape `(n_pool, 128, 10)`. The notebook stitches using
the A001 channel only (`channel=0`) so the HMM comparison stays in the same units as the reference.
Correlation evaluation uses all ten channels from the generated pools.

## Deviations from the reference

Four changes were forced by library versions or by outright bugs. None were discretionary.

**Diffusion uses ten assets with A001 regime labels.** Stage 1 labels regimes from z-scored A001 alone,
matching the reference HMM. Stage 2 attaches nine other ruya assets by date for multi-asset specialist
training and correlation evaluation. The HMM and reference-style plots still consume A001 only.

**Changepoint penalty raised from 10 to 22.** The reference leaves `Vol_Regime.get_changepoints` at its
default penalty of 10, which produced five regimes on its 2022 libraries but produces three on current
ones. The cause is in `self_tuning_spectral_clustering`, which selects a cluster count by strict argmin
of an alignment cost that is not normalized by cluster count. The cost grows with the count, so the
search structurally under-selects, and the reference's five was contingent on its library versions.
Forcing `min_clusters=5` does not help, because one rotated eigenvector column then never wins the
argmax and the result collapses to four.

Sweeping the penalty, the regime count is unstable (5, 4, 4, 5, 5, 4, 5, 5, 2 across 22 to 50), so
several values yield five regimes. The binding constraint turns out to be split coverage rather than
window yield: at `pen=30` and `pen=40` at least one regime falls entirely outside the inner training
slice, so the HMM never observes it and its emission parameters are drawn from the prior instead of
estimated. `pen=22` is the only candidate giving five variance-ordered regimes with all five present in
the training slice, and it reproduces the reference accuracies closely.

**Initial distribution reindexed over all regimes.** The reference builds it with
`value_counts(normalize=True).sort_index()`, which yields a vector as long as the number of regimes
actually present. Because volatility regimes cluster in time, a temporal split can strand one, and the
short vector then fails in `forward_backward`. We reindex over all five.

**Validation regime slicing corrected.** Reference cell 17 writes `vc.regime_labels[train_val_split]`,
a scalar index rather than a slice, so `val_data['regime']` is one value broadcast across every row. We
slice properly. No reported number changes, because the reference recomputes correct validation labels
in its accuracy and backtest cells and never reads `val_data['regime']`.

**Pyro seeded.** The three NumPyro variants take an explicit `PRNGKey`; the reference never seeds Pyro,
so the neural HMM is not reproducible. It is now seeded from `hmm.neural.seed`. This makes a given run
repeatable but does not make the estimator stable; see limitations.

## Limitations

**Diffusion train span is shorter than the HMM train span.** Several assets lack prices before 2001, so
ten-asset windows cover 3395 training days (2001–2014) rather than 6773 (1988–2014). Regime 0 is
especially thin in this overlap: only 130 days and three windows, so stage 3 skips its specialist
unless you retune labeling or accept a four-specialist pool.

**Regime 4 is thin.** The highest-volatility regime covers 255 of 6773 training days across two
segments, roughly one independent 128-day window. Its specialist has very little to learn from and will
largely reproduce its training data. This is a property of the data rather than the method, since
crises are rare, but it means claims about the crisis regime are unsupported.

**The neural HMM is unstable.** Repeated fits of the identical configuration land anywhere between 37%
and 71% train accuracy, because the emitter sometimes collapses the three highest-volatility regimes
onto nearly the same sigma. The ELBO is converged in all cases, so this is initialization sensitivity
rather than an insufficient step budget. Seeding fixes reproducibility, not the underlying variance.
Treat any single neural HMM number as a draw from a wide distribution.

**Labels are not reproducible bit-for-bit** across library versions, for the reason described above.
`scripts/01_label_regimes.py` caches them to `data/processed/regime_labels.npz`, and every downstream
stage reads that cache rather than recomputing, so the specialists and the notebook cannot disagree
about what a regime means. Re-running stage 1 with `--force` invalidates the trained specialists.

**Regime coverage is uneven across the split.** Regimes persist for months, so a temporal split cannot
distribute them evenly and at least one regime is typically absent from the validation slice.
Validation accuracy is measured over fewer than five regimes.

**Windows are stitched independently.** Concatenation introduces a discontinuity every 128 steps, and
nothing enforces continuity across a regime change. The reference GAN shares this artifact, since
`recursive_simulator` also concatenates independent chunks, so it is not a regression, but the
generated series matches per-regime marginals far better than boundary dynamics.

**Global standardization leaks scale.** Log returns are z-scored using statistics of the entire series,
including the held-out portion. This follows the reference and is preserved deliberately.

**Labels are exogenous.** The HMM is fitted against labels from a separate volatility-clustering
pipeline, so its accuracy measures agreement with that pipeline, not recovery of a ground-truth latent
state.

## Reference figures

Accuracies reported by the source notebook, which the pipeline is checked against. They depend on the
regime labels, which cannot be reproduced exactly, so they are a benchmark to land near rather than an
assertion.


| variant          | train  | train top-2 | validation | validation top-2 |
| ---------------- | ------ | ----------- | ---------- | ---------------- |
| supervised       | 82.28% | 98.90%      | 68.48%     | 95.87%           |
| markov switching | 82.00% | 98.92%      | 68.60%     | 95.22%           |
| semi-supervised  | 77.65% | 98.07%      | 68.48%     | 95.75%           |
| neural           | 77.14% | 95.83%      | 62.40%     | 94.98%           |


