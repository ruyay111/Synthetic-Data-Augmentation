# HMM-Diffusion

Regime-conditional synthetic financial time series. Volatility regimes are detected on S&P 500 total
return log returns, one diffusion specialist is trained per regime, and a hidden Markov model stitches
specialist output along an estimated regime path.

## Setup

```
pip install -r requirements.txt
```

## Running

```
python scripts/01_label_regimes.py
python scripts/02_build_diffusion_dataset.py
python scripts/03_train_specialists.py
python scripts/04_generate_pools.py
jupyter lab HMM-Diffusion.ipynb
jupyter lab Downstream-Volatility.ipynb
```

Stage 3 needs a GPU and is the slow step. Stages 1, 2, and 4 run on a laptop.

If stages 3 and 4 are missing, the notebooks use placeholder pools resampled from real per-regime
returns. They still run end to end, but generated-data plots reflect the resampler, not diffusion.

For stage 3:

```
PYTHON=/path/to/python DEVICE=cuda EPOCHS=50 ./scripts/03_train_specialists.sh
REGIMES=0 EPOCHS=2 ./scripts/03_train_specialists.sh
```

## Layout

```
configs/default.yaml       pipeline constants
scripts/                   stages 1–4
src/hmmdiff/               data, regimes, pools, stitching, HMM wrappers, plots
reference_model/diffusion/   DDPM training and sampling
reference_model/hmmgan/      HMM and regime detection
HMM-Diffusion.ipynb          main analysis
Downstream-Volatility.ipynb  volatility forecast on the stitched series
```

## Method

Regime labels come from Vol_Regime on z-scored A001 log returns. The HMM is fit on the inner training
slice (pre-2014). Diffusion specialists train on ten-asset 128-day windows with regime labels mapped
by date. Pools are sampled per regime and stitched using the HMM's estimated path. The downstream
notebook builds a synthetic price series aligned to the 2014+ benchmark window and runs a random-forest
volatility forecast with mixed real and synthetic training data.
