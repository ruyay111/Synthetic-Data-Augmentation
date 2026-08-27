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

One patch, in `run.py`: a `--skip_test` flag.

Upstream unconditionally runs `exp.test(size=500, sample_step=[0.1 ... 1.0], ...)` after training,
which generates 500 windows at each of twelve sample-step fractions. We sample pools separately in
`scripts/04_generate_pools.py` at a single sample step, so for five specialists that sweep is pure
overhead. The flag defaults to `False`, so behaviour without it is identical to upstream.

## `hmmgan/`

Source: `HMM GAN_updated/hmmgan1`, subpackages `hmm/`, `state_estimation/`, and `evaluation/` only.

Excluded: `gan/`, `layers/`, `loss.py`, `functions.py`, `optim.py`, and `utils.py`. Those are the
GAN-specific modules and are the only parts of `hmmgan1` that import TensorFlow. Excluding them keeps
TensorFlow out of this repository's dependency tree entirely.

`utils.hmm_gan_plot` and `utils.recursive_simulator` were needed downstream but live in the excluded
`utils.py`; they are reimplemented without the TensorFlow dependency in `src/hmmdiff/stitch.py`.

No patches. The three subpackages are byte-identical to upstream.
