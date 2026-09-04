# Synthetic Data Augmentation for Portfolio Optimization and Financial Forecasting

Mean-variance optimization and volatility forecasts are unstable when history is short and markets
switch regimes. This repository expands training samples with synthetic return paths from
diffusion models and asks whether that augmentation helps, when regime-conditioning wins, and how
results differ in high- versus low-volatility windows.

Two generators are compared on the same equal-weight (EW) ten-asset panel:

- **Unconditional diffusion** learns one regime-free return distribution.
- **HMM-Diffusion** trains a specialist per volatility regime and, at each hold, samples from the
  specialist for the HMM-forecasted regime.

## Data

Raw prices are stored as CSV.

| File | Source |
| --- | --- |
| `data/raw/benchmark_data.csv` | Ten-asset daily close panel (`as_of` date column; names `A001`, `A004`, `A006`, `A008`, `A009`, `A011`, `A012`, `A013`, `A014`, `A015`). Leading stale rows are dropped via `SKIP_ROWS` in `src/hmmdiff/constants.py`. Calendar overlap used in the EW pipeline is 2001-01-01 to 2022-08-31. |

Processed returns, regime labels, windows, and sampled pools are written under `data/processed/`
and `data/pools/`.


## Method

Both arms start from **equal-weight log returns**: each of ten assets is z-scored, then averaged.
`RegimeProcessor` clusters that series into volatility regimes (2001-01-01 to 2022-08-31).
The HMM trains on 2001-01-01 to 2014-01-03; 2014-01-06 to 2022-08-31 is held out.

- **Offline.** Unconditional diffusion trains on all windows. HMM-Diffusion trains one diffusion
  specialist per regime plus a supervised HMM on the EW train slice.
- **Online (MVO).** A lookback is mixed with synthetic days so the overall synth share is
  0% through 90% (`synth_rows / (real_rows + synth_rows)`). HMM-Diffusion draws from the specialist
  pool for the predicted dominant regime; unconditional diffusion draws from a single pool. Markowitz
  is run on the original ten assets.
- **Volatility forecast.** A forest is trained on mixed real and synthetic rows and evaluated on a
  held-out real test set.


## Package requirements

Install from this file so versions match the environment the pipeline was run in:

```
pip install -r requirements.txt
```

## Running

```
python scripts/01_label_regimes_ew.py
python scripts/02_build_diffusion_dataset_ew.py
python scripts/03_train_specialists_ew.py
python scripts/04_generate_pools_ew.py
jupyter lab HMM-Diffusion-EW.ipynb
jupyter lab hmm-mixed-mvo-ew.ipynb
jupyter lab Downstream-Volatility.ipynb
```

`HMM-Diffusion-EW.ipynb` is the overall workflow notebook (labels, HMM, specialist stitch).
Module notebooks:


`hmm-mixed-mvo-ew.ipynb` is the rolling MVO experiment (Sharpe, Calmar, high-vol vs low-vol).
Unconditional windows come from `trained_diffusion_withoutregime/generated_data.csv`.

## Layout

```
configs/ew.yaml                         EW pipeline YAML
scripts/01–04_*_ew.py                   stages 1–4
src/hmmdiff/constants.py                global constants
src/hmmdiff/data_collection/            data processing classes
src/hmmdiff/model_design/               HMM and diffusion class hierarchy
src/hmmdiff/testing_analytics/          MVO, forecast, and plot classes
reference_model/diffusion/              DDPM training and sampling
reference_model/hmmgan/                 HMM and Vol_Regime
data/raw/benchmark_data.csv             raw ten-asset CSV
HMM-Diffusion-EW.ipynb                  overall EW workflow
hmm-mixed-mvo-ew.ipynb                  MVO backtest
Downstream-Volatility.ipynb             volatility and return R^2 vs mix share
```
