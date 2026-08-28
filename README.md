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

Stage 3 needs a GPU for training the diffusion model, which is computational heavy.

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

Regime labels come from Vol_Regime on z-scored A001 log returns. The HMM is fit on the training
data (pre-2014). Diffusion specialists train on ten-asset 128-day windows with regime labels.
Pools are sampled per regime and stitched using the HMM's estimated path. The downstream
notebook builds a synthetic price series aligned to the 2014+ benchmark window and runs a random-forest
volatility forecast with mixed real and synthetic training data.
