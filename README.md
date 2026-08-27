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
python scripts/01_label_regimes.py            # preprocess, detect regimes, cache labels
python scripts/02_build_diffusion_dataset.py  # cut per-regime 128-day training windows
./scripts/03_train_specialists.sh             # train 5 specialists; needs a GPU
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

Prices are S&P 500 total return closes (`A001`) from 1988-01-19, converted to log returns and z-scored
using the mean and standard deviation of the whole series. The 9031 returns are split 75/25 into 6773
train and 2258 test; the train portion is split 75/25 again into 5079 inner-training and 1694
validation days. The test portion is never used.

Regimes come from `Vol_Regime`: GARCH(1,1) conditional volatility, PELT changepoint detection, a
Wasserstein affinity between segments, then self-tuning spectral clustering. Labels are reordered so
regime index increases with variance.

Specialists are unconditional DDPMs trained on 128-day windows cut from contiguous same-regime runs of
the full 6773-day training series. Windows are z-scored log returns with `enc_in=1`. Segments shorter
than 128 days are cyclically tiled rather than dropped.

Sampling produces `n_pool` windows per regime, inverse-transformed back to z-scored log-return units.
Stitching walks a regime path and takes the next unused value from that regime's pool.

## Deviations from the reference

Four changes were forced by library versions or by outright bugs. None were discretionary.

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

**`Corr` loss term dropped.** The reference diffusion configuration uses
`1.0-KL2_N+1.0-Corr+1.0-FFT`. `Corr_Loss` matches the cross-channel correlation matrix by reducing its
strictly-upper triangle, which for univariate data is a 1×1 matrix with an empty upper triangle. The
mean over an empty tensor is NaN on CPU and CUDA. `Loss_Wrapper` sums terms without sanitizing, so the
reported loss is NaN every epoch and the loss curve is meaningless. Gradients happen to survive,
because backward through an empty tensor contributes nothing, so the model still trains on the
remaining terms while reporting nothing usable. The term is structurally vacuous at `enc_in=1`, not
merely unhelpful, so it is removed. The configuration is `1.0-KL2_N+1.0-FFT`.

Worth noting for anyone smoke-testing on a Mac: MPS returns 0.0 rather than NaN for a mean over an
empty tensor, so this bug is invisible on Apple Silicon and only appears on CUDA.

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

| variant | train | train top-2 | validation | validation top-2 |
|---|---|---|---|---|
| supervised | 82.28% | 98.90% | 68.48% | 95.87% |
| markov switching | 82.00% | 98.92% | 68.60% | 95.22% |
| semi-supervised | 77.65% | 98.07% | 68.48% | 95.75% |
| neural | 77.14% | 95.83% | 62.40% | 94.98% |
