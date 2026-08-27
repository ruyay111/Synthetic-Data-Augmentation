# HMM-Diffusion: implementation plan

## Objective

Reproduce the pipeline and visualizations of `HMM GAN_updated/Supervised HMMs.ipynb` exactly, with the
per-regime GAN generators replaced by per-regime ("specialist") diffusion models. Same input data
(`Copy of benchmark_data.csv`, column `A001`), same regime labeling, same four supervised-HMM variants,
same stitching rule, same plots and metrics. The generative model is the only component that changes.

The repository must be self-contained: a fresh clone plus `pip install -r requirements.txt` plus three
scripts must be enough to run the deliverable notebook end to end.

## Decisions (resolved)

| Question | Decision |
| --- | --- |
| Channels | Univariate, `enc_in=1` on `A001` only, matching the GAN's `data_channel=1` |
| HMM variants | All four: supervised HMM, Markov switching model, semi-supervised HMM, neural HMM |
| Compute | CUDA; scripts default to `--device cuda` with an override flag |
| Training slice | Reproduce the reference: specialists train on the full 6773-day train split |
| GAN baseline | Diffusion only; no TensorFlow anywhere in the repo |
| Vendoring | Copy the diffusion and HMM source into the repo, plus the raw CSV |
| Deliverable | `notebooks/HMM-Diffusion.ipynb` |
| CUDA workflow | Same repo cloned on the CUDA box; scripts 01-04 run there start to finish |
| Pool size | `n_pool=128` per regime (16384 samples), so pools can be resampled without regenerating |
| Dependencies | Loose bounds in `requirements.txt`, plus the exact ruya CUDA 12.4 freeze committed as reference |

Assumptions I am making on the smaller points, stated so you can override:

- HMM source of truth is `hmmgan1` (`hmm/`, `state_estimation/`, `evaluation/`), not the ruya
  `train/hmm` port. Reproducing all four variants requires `markov_switching_model`,
  `semi_supervised_hmm`, `neural_hmm`, and `Emitter`, which exist only in `hmmgan1`. Those three
  subpackages are TensorFlow-free (verified: only `loss.py` and `functions.py` import TF, and neither
  is needed).
- Pools are plain concatenations of independent 128-day windows, reproducing the seam artifact the GAN
  already has via `recursive_simulator`.
- The cell-17 slicing bug is fixed and the fix is noted. It changes no reported number.
- No extra statistical diagnostics beyond the notebook's own plot set.

## Verified facts about the reference pipeline

Established by reading the notebook source and running the preprocessing:

- Input: `Copy of benchmark_data.csv`, indexed by `as_of`, column `A001` (S&P 500 TR USD).
- Preprocessing: drop first 2100 rows, `log(close).diff().dropna()`, then z-score using the mean and
  std of the **entire** post-1988 return series, computed before any split.
- Series length 9031 days, 1988-01-19 to 2022-08-31.
- Outer split 75/25: train 6773 days (to ~2005-12-15), test 2258 days.
- `Vol_Regime` is fit on the 6773-day train series only, `max_clusters=5`: `get_vol()` (ARCH
  conditional volatility) → `get_changepoints()` (PELT, RBF kernel, `pen=10`) → `get_attr()`
  (Wasserstein distances between changepoint segments, self-tuning affinity with 8th-NN bandwidth) →
  `assign_clusters(max_clusters=5)` (self-tuning spectral clustering on segments, relabeled so regime 0
  has the lowest variance).
- Inner split of the train series, 75/25: `train_data` 5079 rows, `val_data` 1694 rows, columns
  `regime`, `emission`, `emission_lag`. The notebook's "test" evaluation runs on `val_data`, not on the
  2258-day outer test set.
- Generation (cell 14): `steps = round(6773 / 128) = 53` chunks of 128 per regime, so each regime's
  pool is a flat array of `53 × 128 = 6784` values in z-scored log-return units.
- Stitching (`hmm_gan_plot`, cells 15 and 34): walk the regime path; at each step pop the next unused
  value from that regime's pool via a per-regime counter. Pools must be at least as long as the number
  of steps assigned to that regime.
- GAN reference config: `data_len=128`, `data_channel=1`, univariate.

Per-variant hyperparameters, all sharing the same evaluation and plots:

| Variant | Fitting | Key settings |
| --- | --- | --- |
| Supervised HMM | NumPyro NUTS | `num_warmup=200`, `num_samples=100`, `num_chains=1`, `PRNGKey(1)` |
| Markov switching | NumPyro NUTS | same, on `train_data.values[1:]` to drop the lag NaN |
| Semi-supervised HMM | NumPyro NUTS | same, 50/50 labeled/unlabeled split of the train slice |
| Neural HMM | Pyro SVI | `Emitter(z_dim=1, hidden_dim=10, emission_dim=1)`, `AutoDelta`, `TraceEnum_ELBO`, `lr=0.1`, 100 steps |

All four then run the same `forward_backward(init_dist, observations, transmat, mu, sigma, 5)`, report
`accuracy_score` and `top_k_accuracy_score(k=2)` on train and val, plot estimates against truth, call
the two-panel generated-returns figure, and print the per-regime real-vs-generated mean and variance
table.

## Repository layout

```
HMM-Diffusion/
  README.md
  PLAN.md
  requirements.txt                  # loose bounds, works on CUDA and Apple Silicon
  requirements-ruya-freeze.txt      # exact CUDA 12.4 versions, reference only
  configs/default.yaml              # paths, splits, n_regimes, seq_len, sampling params
  data/
    raw/benchmark_data.csv          # copied from HMM GAN_updated/Data/Copy of benchmark_data.csv
    processed/
      sp500tr_returns.parquet
      regime_labels.npz             # labels, changepoints, clusters, conversion_dict, metadata
      regime_windows/regime_{k}.npy # (N_k, 128, 1)
      regime_windows/manifest.json
    pools/regime_k{k}/windows.npy   # (128, 128, 1), z-scored log-return units
    pools/regime_k{k}/meta.json
  src/hmmdiff/
    data.py        # preprocessing + splits (notebook cells 4-5)
    regimes.py     # Vol_Regime wrapper + label caching (cell 6)
    windows.py     # regime runs -> (N,128,1) training windows
    pools.py       # load pools, flatten to generated_images dict
    stitch.py      # TF-free hmm_gan_plot equivalent + backtest table
    models.py      # thin wrappers around the four HMM variants
    plots.py       # all notebook figures
  third_party/
    diffusion/     # vendored ruya train/diffusion (run.py, src/)
    hmmgan/        # vendored hmmgan1 hmm/, state_estimation/, evaluation/
  scripts/
    01_label_regimes.py
    02_build_diffusion_dataset.py
    03_train_specialists.sh
    04_generate_pools.py
  notebooks/HMM-Diffusion.ipynb
```

`third_party/hmmgan/` excludes `gan/`, `layers/`, `loss.py`, `functions.py`, `optim.py`, and `utils.py`
— everything that imports TensorFlow. `hmm_gan_plot` and `recursive_simulator` from `utils.py` are
reimplemented in `src/hmmdiff/stitch.py` without the TF dependency.

## Stage 1 — script 01: label the regimes

`scripts/01_label_regimes.py` covers notebook cells 4 through 10.

Reads `data/raw/benchmark_data.csv`, applies the exact preprocessing chain (`iloc[2100:]`, log diff,
global z-score, 75/25 outer split, 75/25 inner split), writes `sp500tr_returns.parquet` with columns
`date`, `log_return`, `z_return`, `split`.

Runs `Vol_Regime` on the 6773-day train series with `max_clusters=5` and caches `regime_labels`,
`changepoints`, `clusters`, and `conversion_dict` to `regime_labels.npz`. Caching is required, not an
optimization: `assign_clusters` runs a conjugate-gradient rotation search whose result is not
guaranteed stable across runs, and every downstream stage plus the notebook must read identical labels.
Re-running the script without `--force` reuses the cache.

Validation written into the cache metadata: 5 regimes present, label array length 6773, per-regime
variance monotonically increasing in the label index, changepoint count, and average regime length.

## Stage 2 — script 02: build the trainable diffusion dataset

`scripts/02_build_diffusion_dataset.py`, adapted from `train/scripts/build_regime_window_datasets.py`
for univariate input.

Finds contiguous runs of identical regime label in the 6773-day train series. Runs of length >= 128
yield sliding windows with `stride=1`; shorter runs are tiled cyclically so no regime is dropped. Saves
`(N_k, 128, 1)` float arrays plus a manifest with per-regime window and segment counts.

The manifest is the diagnostic that matters here. With 5 regimes over 6773 days, the sparse
high-volatility regimes may yield few genuinely independent source segments; if a regime has fewer than
roughly 50 distinct segments its specialist will largely memorize, and that belongs in the limitations
section rather than being papered over. For scale, the ruya 10-asset manifest on a 5652-day panel gives
1861 / 777 / 233 / 446 / 303 windows across regimes 0-4.

## Stage 3 — script 03: train five specialists

`third_party/diffusion/` is a verbatim copy of the ruya `train/diffusion` tree.
`scripts/03_train_specialists.sh` is derived from their `train_specialist_diffusions.sh` with three
changes: `--enc_in 1`, `--data_path` pointing at our univariate windows, and a `DEVICE` environment
override defaulting to `cuda`. Everything else stays at the tuned ruya settings:

```
--task_name diffusion_denoised_x --model UniTST_MP --data RegimeWindows
--seq_len 128 --enc_in 1 --batch_size 32 --sample_multiplier 8 --train_epochs 50
--causal_mask --ind_proj --RoPE --channel_embed
--learning_rate 0.0027983303288563873 --lr_decay_rounds 10
--loss "1.0-KL2_N+1.0-Corr+1.0-FFT"
--description "specialist_regime_{k}"
```

Units work out without extra rescaling: `Dataset_RegimeWindows` fits a
`QuantileTransformer(output_distribution='normal')` on the flattened windows and `generate_data`
applies `inverse_transform`, so a model trained on z-scored input emits z-scored output, directly
comparable to the notebook's `emission` column.

One thing to check on the first run: the `Corr` loss term is a cross-channel correlation loss and is
degenerate with `enc_in=1`. If it produces NaNs, fall back to `"1.0-KL2_N+1.0-FFT"` and record the
change in the README.

## Stage 4 — script 04: sample the regime pools

`scripts/04_generate_pools.py`, adapted from `generate_specialist_pools.py` with `--n-assets 1` and
`--n-pool 128`. That gives 16384 samples per regime, well above the notebook's 6784 and above the
maximum possible per-regime demand of 6773, leaving enough headroom to draw alternative pool
realizations without re-running the sampler. Keeps `--sample-step 450`,
`--temperature 0.7473231454237225`, `sampler="DDPM"`, `method="discrete"`.

Writes `data/pools/regime_k{k}/windows.npy` shaped `(128, 128, 1)` plus `meta.json`.
`src/hmmdiff/pools.py` flattens each pool to a 1-D array of length 16384 and exposes it as
`generated_images[str(k)]`, matching the shape and dtype the notebook's stitching code expects. It also
takes a seed so a subset of windows can be drawn reproducibly for sensitivity checks.

Key sanity check before proceeding: per-regime pool variance must follow the same ordering as the real
per-regime variance, with regime 0 lowest. This is the test of whether specialization actually happened.

## Stage 5 — the notebook

`notebooks/HMM-Diffusion.ipynb` mirrors the reference notebook section for section, calling
`src/hmmdiff` functions rather than inlining logic, and loading the cached artifacts from scripts 01-04
rather than recomputing them. Structure:

Setup and data. Load the returns parquet and the cached regime labels. Reproduce the close-price plot,
the returns-with-regime-spans plot, the EWMA volatility overlay, the regime-length statistics, and the
regime label series (cells 4-10).

Generated data. Load the five pools into `generated_images`, then reproduce the oracle stitch of
cell 15 using the true regime labels.

Preprocessing for modeling. Build `train_data` and `val_data` with `regime`, `emission`, `emission_lag`;
print the per-regime empirical moments, the regime-length table, and the empirical and outbound
transition matrices (cells 17-21).

Four model sections, one per variant, each following the reference structure exactly: fit, extract
posterior parameters, `forward_backward` on train, train accuracy and top-2 accuracy, estimates-vs-truth
plot, two-panel generated-returns figure, then the same four steps on the validation slice, then the
per-regime real-vs-generated mean and variance table.

The two-panel figure keeps the reference layout, `ylim(-8, 6)`, and axes, retitled "HMM Diffusion
Generated Returns".

## Known divergences from the reference notebook

Stated in the notebook rather than hidden. The four entries below marked "resolved during
implementation" were not anticipated when this plan was written; `README.md` carries the full
reasoning for each.

**Changepoint penalty 10 to 22 (resolved during implementation).** The reference default no longer
yields five regimes on current libraries, because the cluster-count search minimizes an
un-normalized alignment cost and therefore under-selects. Several penalties give five regimes, but
only `pen=22` also places all five in the inner training slice, which the HMM requires in order to
estimate every emission rather than fall back on the prior. It reproduces the reference accuracies
closely.

**`Corr` loss term dropped (resolved during implementation).** Anticipated below as a risk and
confirmed: at `enc_in=1` it reduces an empty upper triangle, and the mean over an empty tensor is NaN
on CPU and CUDA. The loss is now `1.0-KL2_N+1.0-FFT`. MPS returns 0.0 instead of NaN, so the bug is
invisible on Apple Silicon.

**Initial distribution reindexed (resolved during implementation).** The reference builds it from
`value_counts`, which yields a short vector when a regime is absent from a slice and then fails in
`forward_backward`. Reindexed over all five regimes.

**Pyro seeded (resolved during implementation).** The reference never seeds Pyro, so the neural HMM
is not reproducible. Seeding makes a run repeatable but does not stabilize the estimator; see the
risks section.

Diffusion training data is windowed from contiguous same-regime runs with cyclic tiling for short runs.
The GAN's per-regime training procedure is not reproducible from what survives on disk (only weights and
a hyperparameter table), so the two generators are not matched on training-set construction.

`Dataset_RegimeWindows.__getitem__` ignores its index and returns a uniformly random window, with
`sample_multiplier=8` inflating epoch length. Epochs are random draws with replacement, not passes over
the data. This is the ruya convention, kept for consistency with their tuned hyperparameters.

Concatenating independent 128-day windows introduces a discontinuity every 128 steps. The GAN pipeline
has the identical artifact, so plain concatenation is the faithful choice.

The reference notebook fits `Vol_Regime` on the full 6773-day train series and trains its generators on
the same span, then evaluates on a 1694-day slice carved out of it. This is mild leakage inherited from
the reference and reproduced deliberately.

Cell 17 of the reference has a bug: `val_regimes = vc.regime_labels[train_val_split]` is a scalar index,
not a slice, so `val_data['regime']` is a constant broadcast across all 1694 rows. It affects no
reported number, because cell 34 overwrites `results['regime']` with the correct slice and the accuracy
calls use that slice. We fix it and note the fix.

## Validation checkpoints

1. Preprocessing: series length 9031, splits 6773 / 2258 and 5079 / 1694, dates 1988-01-19 to 2022-08-31.
2. Regime labels: 5 regimes, variance monotone in label index, changepoint and average-length statistics recorded.
3. Windows: manifest shows non-empty windows for all 5 regimes; per-regime empirical mean and variance of the windowed data match the corresponding slice of the source series.
4. Specialists: per-regime training loss curves saved, no NaN losses.
5. Pools: shape `(128, 128, 1)`, finite, per-regime variance ordering matches the real per-regime ordering.
6. Pool sufficiency: for every regime, pool length >= steps assigned to that regime in both the train and val regime paths.
7. HMM: train and val accuracy within a small tolerance of the reference notebook's stored outputs, for all four variants. These depend only on the HMM and the labels, not on the generator, so a mismatch means the regime labels diverged and Stage 1 needs revisiting before any diffusion result is meaningful.
8. Final figures: same axes, limits, and titles as the reference.

## Build order

1. Repo skeleton, `requirements.txt`, vendored `third_party/`, raw CSV copy.
2. Script 01 and `src/hmmdiff/data.py`, `regimes.py`. Verify checkpoints 1 and 2.
3. Script 02 and `src/hmmdiff/windows.py`. Verify checkpoint 3.
4. Script 03. Smoke-test regime 0 for one or two epochs before launching all five.
5. Script 04 and `src/hmmdiff/pools.py`. Verify checkpoints 5 and 6.
6. `src/hmmdiff/models.py`, `stitch.py`, `plots.py`, then the notebook. Verify checkpoints 7 and 8.

Stages 2 and 6 can proceed against a placeholder pool of resampled real per-regime returns, so the
notebook and plotting code can be finished and validated while the specialists train.

## Risks to watch

Regime data volume. Confirmed and unresolved: regime 4 has 255 days across two segments, roughly one
independent 128-day window. Script 02 reports independent (non-overlapping) window counts rather than
raw window counts, since stride-1 sliding windows overlap almost completely and the raw count
overstates how much distinct data a specialist sees. This belongs in the limitations and does.

Label instability. `assign_clusters` is not deterministic across runs, so the cached labels from script
01 are the single source of truth for scripts 02-04 and the notebook. Any stage that re-derives labels
instead of reading the cache will silently desynchronize from the trained specialists.

The `Corr` loss term with `enc_in=1`. Confirmed and resolved by dropping the term; see divergences.

Neural HMM instability. Surfaced during implementation: repeated fits of the identical configuration
land anywhere between 37% and 71% train accuracy, because the emitter sometimes collapses the three
highest-volatility regimes onto one sigma. The ELBO is converged in every case, so this is
initialization sensitivity, not step budget. Seeding fixes reproducibility only.

Dependency resolution. The stack spans torch, NumPyro/JAX/funsor, and Pyro simultaneously. JAX and torch
coexisting on one CUDA box is the most likely install friction point; the committed ruya freeze is the
fallback reference if loose bounds resolve badly.
