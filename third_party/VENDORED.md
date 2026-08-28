# Vendored third-party source

Both trees are copied in so that a fresh clone of this repository can run the full pipeline without
access to the original working directories.

## `diffusion/`

Source: `ruya/Synthetic-data-tests/train/diffusion` (`run.py` and `src/`).

Excluded from the copy: `checkpoints/`, `test_results/`, `__pycache__/`, and the shell scripts, which
are replaced by `scripts/03_train_specialists.sh` and `scripts/04_generate_pools.py` in this repo.

There are no `__init__.py` files anywhere under `src/`; the upstream tree relies on implicit namespace
packages and on callers putting the diffusion directory on `sys.path`. The copy preserves this.

`run.py` chdirs to its own directory on import, so relative `--data_path`, `--checkpoints`, and
`--test_results` arguments resolve against `third_party/diffusion/`, not the caller's working
directory. Our scripts always pass absolute paths.

### Local patches

Two patches, both in `run.py` plus a small helper in `src/exp/exp_basic_diffusion.py`.

`--skip_test` skips the post-train evaluation sweep. Upstream unconditionally runs
`exp.test(size=500, sample_step=[0.1 ... 1.0], ...)`, which generates 500 windows at each of twelve
sample-step fractions. `--eval_sample_step` replaces that sweep; values greater than 1 are absolute
step counts so `450` is reverse-diffusion length 450, not `total_steps * 450`. Stage 3 passes
`sampling.sample_step` from `configs/default.yaml` (450) and writes dist/autocorr/moments/cov/corr
plots under `test_results/<run>/450_discrete_DDPM_<temp>/`. The flag defaults keep upstream behaviour
when neither `--skip_test` nor `--eval_sample_step` is set.

## `hmmgan/`

Source: `HMM GAN_updated/hmmgan1`, subpackages `hmm/`, `state_estimation/`, and `evaluation/` only.

Excluded: `gan/`, `layers/`, `loss.py`, `functions.py`, `optim.py`, and `utils.py`. Those are the
GAN-specific modules and are the only parts of `hmmgan1` that import TensorFlow. Excluding them keeps
TensorFlow out of this repository's dependency tree entirely.

`utils.hmm_gan_plot` and `utils.recursive_simulator` were needed downstream but live in the excluded
`utils.py`; they are reimplemented without the TensorFlow dependency in `src/hmmdiff/stitch.py`.

No patches. The three subpackages are byte-identical to upstream.
