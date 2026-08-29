# Plan: HMM-mixed MVO notebook

This is a design document for a new notebook `hmm-mixed-mvo.ipynb`. Do not implement until this plan is approved.

The notebook compares two 10-asset mean-variance backtests on the existing HMM-Diffusion validation window. For `n ≥ 1`, **both** methods add synthetic data on top of the same real lookback:

- **hmm-diffusion**: 60-day real lookback plus `n` sampled 60×10 specialist windows from the predicted regime pool.
- **mixed**: 60-day real lookback plus `n` sampled 60×10 windows from the regime-free diffusion file `trained_diffusion_withoutregime/generated_data.csv`.

Do not use calendar date-matched synth. The ruya calendar mixed arm is out of scope.

Regime forecasting on the hold uses the transition matrix only. Emissions are applied only after the hold is scored, and only on those 60 days.

## Goal

1. Fit the **supervised HMM only**, on this repo’s inner-train slice.
2. On the validation window, produce a **four-line regime plot** (daily, same style as `plots.plot_estimates` in `HMM-Diffusion.ipynb`).
3. Run a **non-overlapping 60-day MVO** on that window, sweeping the number of synthetic series like Downstream-Volatility.
4. Report Sharpe and Calmar by **high vol / low vol**, labeled from **true clustering regimes**, not calm/crisis calendars.

## Splits (from `HMM-Diffusion.ipynb`)

The return series is A001 log returns, 1988-01-20 to 2022-08-31, 9,031 days. Splits are `train_fraction=0.75` then `train_val_fraction=0.75`.

| Slice | Days | Dates |
| --- | ---: | --- |
| Full series | 9,031 | 1988-01-20 → 2022-08-31 |
| Outer train | 6,773 | 1988-01-20 → 2014-01-03 |
| **Inner train (HMM fit)** | **5,079** | **1988-01-20 → 2007-07-09** |
| **Validation (eval)** | **1,694** | **2007-07-10 → 2014-01-03** |
| Outer test | 2,258 | 2014-01-06 → 2022-08-31 |

HMM training uses **inner train only** (through 2007-07-09). Rolling MVO and the four-line plot use **validation only** (2007-07-10 through 2014-01-03). That is the current notebook val end date.

The 10-asset panel is shorter than A001 because some names start later. Align MVO to dates that exist in both the univariate HMM series and the 10-asset simple-return panel. Drop a window if the 60-day lookback or 60-day hold is incomplete.

## Constants

```
HORIZON = 60
STEP = 60
LOOKBACK = 60
EVAL_START = validation start  # 2007-07-10
EVAL_END   = validation end    # 2014-01-03
MIX_GRID   = list(range(0, 11))  # n_synth = 0..10 (11 points, same count as downstream 0,10,...,100)
RANDOM_STATE = 42
N_REGIMES = 5
TOP_K = 2
ASSET_COLS = cfg["data"]["asset_columns"]  # A001, A004, A006, A008, A009, A011–A015
HIGH_VOL_REGIMES = {3, 4}
LOW_VOL_REGIMES  = {0, 1, 2}
```

`n` is the number of synthetic 60×10 windows, not a row-mix percent.

- `n = 0`: real lookback only (`t-60` … `t-1`, 60 × 10).
- `n = 1`: that real block plus **one** sampled 60 × 10 synthetic window.

No max-weight cap, no stress ranking, no long-only constraint. Markowitz is sum-to-1 only; shorts allowed. Keep a small ridge on the covariance for numerical stability.

## Emission and HMM

Use this repo’s HMM emission: **A001 z-scored log return** (`returns["z_return"]` / `train_data.emission`), the same series as `HMM-Diffusion.ipynb`. Do not switch to EW-vol ARCH.

Labels: cached Vol_Regime clustering from `scripts/01_label_regimes.py` (`data/processed/regime_labels.npz`).

Fit: `models.fit_supervised_hmm(train_data, N_REGIMES, cfg)` on inner train only. Store `P = fit.transmat`, emission `(μ, σ)`, and `init_dist` from inner-train regime frequencies.

## Rolling regime protocol

Index `t` is the first day of a 60-day hold. Holds are `[t, t+59]`, then the next hold starts at `t+60`.

### Before the first hold

Run a **causal forward filter** (emission updates, no future smoothing) on inner-train emissions through the last inner-train day (`t-1` for the first val window). That filtered `π_{t-1}` is the state entering validation.

Do not use full-sample forward-backward for this filter. Forward-backward on the whole val slice is a **separate diagnostic line** (line 2 below).

### During hold `[t, t+59]` (open loop)

No emission updates inside the hold.

```
π_t     = π_{t-1} @ P
π_{t+1} = π_t @ P
...
π_{t+59} = π_{t+58} @ P
```

Normalize after each multiply. This is `forecast_regime_probability_path(π_{t-1}, P, 60)`.

MVO for this hold is computed **before** the post-hold emission update, using only these open-loop probabilities.

### After the hold is scored

Now the 60 emissions `e_t … e_{t+59}` are observed. Apply the emission update **only on those 60 days**, starting from `π_{t-1}` (the filtered state before the hold). Do not use days `t+60` and later.

The last filtered state, `π_{t+59}^{filtered}`, is the handoff to the next window:

```
π_{t+60} = π_{t+59}^{filtered} @ P
```

Then open-loop again through `t+119`.

### Filter vs smoother

| Quantity | Algorithm | Uses hold emissions? | Used for |
| --- | --- | --- | --- |
| Line 2 (FB estimate) | Full forward-backward on the val slice, same as `evaluate_variant` | Yes, whole val | Plot / accuracy only |
| Open-loop `π` on a hold | `π @ P` only | No | Plot lines 3–4, MVO `k*` |
| Post-hold filter | Forward filter on the just-finished 60 days | Yes, current hold only | Next window’s `π_{t-1}` |

## Four-line regime plot (validation only)

Same step-plot style as `Supervised HMM: Validation` in `HMM-Diffusion.ipynb` (`plots.plot_estimates`), but four series on **val dates only**:

1. **Truth** — daily clustering labels.
2. **Forward-backward estimate** — daily argmax of smoothed `π` from `models.estimate_states` on the val emissions, same call as `evaluate_variant`.
3. **Transition daily argmax** — for each val day, argmax of that day’s open-loop `π` (transition matrix only).
4. **Transition window argmax** — for each 60-day hold, convert daily transition-only hard labels to one-hot, average the 60 one-hots, take argmax, and **repeat that regime for all 60 days** (piecewise constant).

Accuracy, same reporting as `HMM-Diffusion.ipynb` (`models.accuracy_report`, top-1 and top-2):

- Daily: truth vs FB estimate (line 1 vs 2).
- Daily: truth vs transition daily argmax (line 1 vs 3).
- Window-level: window truth = one-hot average of the 60 true labels, then argmax; compare to line 4. Also report top-2 on the averaged one-hot scores if we keep the averaged vector as `y_score`.

If a regime is missing on val (inner train historically lacks some val regimes, and val has lacked regime 0), print the same coverage note as the main notebook and compute accuracy on observed labels.

## MVO, one hold at `t`

### Predicted pool regime `k*`

Causal, from the open-loop hold path only:

- Daily hard label `ŝ_h = argmax π_{t+h}` for `h = 0..59`.
- **MVO `k*` is the window occupancy average:** convert the 60 daily hard labels to one-hot, average, then `k* = argmax`.
- Sample synthetic paths from **that one regime’s pool**, not day-by-day across pools.

### hmm-diffusion training matrix

- Real: 10-asset **simple** returns on `[t-60, t)` (`np.expm1` of log returns). Shape 60 × 10.
- Synthetic: `n` contiguous 60-day slices from `pool_{k*}`. Pools are `(n_pool, 128, 10)` **raw log returns**. Convert with `np.expm1`.
- Column-stack as in ruya `mix_train_with_regime_paths`: real 60 × 10 plus, for each asset, `n` extra columns from the synthetic paths. Then Markowitz on the expanded matrix and `collapse_weights` back to 10 assets.

Every synthetic source is converted to the **same simple-return units** as this real lookback before it enters Markowitz. See **Scale check**.

### mixed training matrix

Same hold, same real lookback `[t-60, t)`. Synthetic paths come from `trained_diffusion_withoutregime/generated_data.csv`, not from `k*` pools and not from calendar dates.

File facts (checked):

- 131,072 rows × 10 columns (`0`–`9`), no date index.
- 131,072 ÷ 128 = **1,024** windows of shape `(128, 10)`.
- Values are **raw log returns** (daily std ~0.001–0.021, not z-scores of std ~1). Same inverse-scaled units as specialist `generated_data.csv` / `data/pools`.

Load by reshaping to `(1024, 128, 10)`. For each hold, draw `n` windows (seed `RANDOM_STATE + t + n`, without replacement if `n ≤ 1024`), take a contiguous 60-day slice, `np.expm1`, then column-stack with the real lookback using the same `mix_train_with_regime_paths` helper as hmm-diffusion.

This arm does **not** use the HMM or regime pools. The only difference from hmm-diffusion is where the `n` synthetic 60×10 windows come from.

### Scale check (all synthetic vs real MVO inputs)

MVO always sees **simple returns**. Before the backtest, print a per-asset table that compares:

| Source | Native units | Converted for MVO |
| --- | --- | --- |
| Real 10-asset lookback panel | raw log returns | `expm1` |
| hmm-diffusion pools `data/pools/regime_k{k}/windows.npy` | raw log returns | `expm1` of 60-day slices |
| mixed `trained_diffusion_withoutregime/generated_data.csv` | raw log returns | `expm1` of 60-day slices |

For each source and each of the 10 assets, report mean, std, min, max, and 1%/99% quantiles on the converted simple returns. Real reference = the 10-asset panel on the MVO eval dates that actually enter lookbacks (val window plus the 60 days before the first hold).

Pass / fail:

- Native synth must look like **daily log returns** (std on the order of 0.001–0.03), not z-scored emissions (std ~1) and not prices.
- After `expm1`, synth std should be within a small factor of the real asset’s std (flag if any asset’s synth std is `< 0.1×` or `> 10×` the real std, or if `|mean|` is implausibly large).
- Spot-check: `generated_data.csv` vs real train overlap 2001-01-01 → 2014-01-03 is already in the same decade (A001 log std 0.017 vs 0.013; other assets similar). Means on the mixed file run a bit high vs real; that is a model bias to report, not a unit mismatch.
- Do not mix z-scored HMM emissions into the MVO matrix.

If a source fails the unit check, stop before the backtest and print which file is on the wrong scale.

### Evaluation

Fix weights at `t`. Apply them to realized 10-asset simple returns on `[t, t+60)`. Compute Sharpe and Calmar with the same formulas as ruya `summarize_portfolio` (annualize with 252).

### High vol / low vol bucket

No calm/crisis calendars. Use **true clustering labels** on the hold:

- A day is **high vol** if true regime ∈ {3, 4}, else **low vol**.
- A window is **high vol** if at least half of its 60 true days are high vol; otherwise **low vol**.

Boxplots: Sharpe and Calmar, hmm-diffusion vs mixed, grouped by high vol / low vol. One figure per `n`, or a small-multiples grid over `n`, in the same boxplot layout as the ruya Sharpe/Calmar figure.

Sweep `n ∈ {0,1,…,10}`. Both methods share the same holds. `n = 0` is real-only for both (they should match up to numerics). For `n ≥ 1`, both add `n` synthetic 60×10 windows to the same real lookback.

## Layout to add (after approval)

```
hmm-mixed-mvo.ipynb          orchestration, plots, tables
src/hmmdiff/mvo/
  __init__.py
  portfolio_core.py          port: mean_var_weights, collapse_weights,
                             mix_train_with_regime_paths,
                             summarize_portfolio, sharpe, calmar
                             (no classify_regime / crisis windows /
                             calendar mix_train_matrix)
  hmm_forecast.py            open-loop P path; post-hold emission filter
  specialist_sample.py       draw 60×10 from pool_k (log → simple)
  mixed_sample.py            reshape generated_data.csv; draw 60×10 (log → simple)
  scale_check.py             per-asset mean/std/quantiles, synth vs real simple returns
  backtest.py                rolling 60/60 loop, hmm-diffusion vs mixed, mix grid
```

Port only what this notebook needs from `/Users/mohanyang/Desktop/ruya/Synthetic-data-tests/evaluation/mvo` and `evaluation/utils/portfolio_core.py`. Adapt imports to `hmmdiff`. Do not copy shock tests, EW-vol HMM, or calm/crisis helpers.

Reuse existing repo code: `hmmdiff.models`, `hmmdiff.data`, `hmmdiff.pools.load_pool`, `hmmdiff.plots.plot_estimates` (extend or wrap for four lines).

## Notebook sections

1. **Setup** — config, constants, load returns/labels/pools/10-asset panel/`generated_data.csv`, print split dates.
2. **Fit supervised HMM** — inner train; print `P` and emission summary.
3. **Rolling open-loop forecast** — walk val in steps of 60; store daily open-loop `π`, daily argmax, window one-hot-average argmax, and post-hold filtered `π_{t+59}`.
4. **Regime plot and accuracy** — four lines; top-1 / top-2 tables.
5. **Scale check** — real vs specialist pools vs mixed CSV, after conversion to simple returns.
6. **MVO backtest** — hmm-diffusion vs mixed over `MIX_GRID`.
7. **Sharpe / Calmar** — boxplots by high vol / low vol; summary table (mean/median/count).

## Causal rules (must hold)

- MVO at `t` may use real returns through `t-1` and open-loop `π` from `π_{t-1}` and `P`.
- MVO at `t` may **not** use hold emissions `e_t … e_{t+59}` or any later day.
- After scoring the hold, update `π` with those 60 emissions only; that update is for the **next** hold.
- Line 2 (forward-backward on all val) is a diagnostic overlay. It is not an input to MVO.

## Validation after implementation

- Inner-train / val dates match the table above.
- First hold starts 2007-07-10 (or the first val day with a full 60-day lookback and 10-asset coverage).
- Open-loop `π` on a hold does not depend on that hold’s emissions (spot-check by perturbing a hold emission and confirming `k*` unchanged).
- Post-hold filter **does** change `π_{t+59}` and the next hold’s starting forecast.
- `n = 0` hmm-diffusion and mixed Sharpe/Calmar match.
- For `n ≥ 1`, both arms add synthetic 60×10 windows; they differ only in the synth source (`pool_{k*}` vs `generated_data.csv`).
- Scale check runs before the backtest; both synth sources are `expm1` of log returns, not z-scores.
- Mixed samples from `trained_diffusion_withoutregime/generated_data.csv` reshaped to `(1024, 128, 10)` and does not use `k*`.
- Four-line plot covers val only; line 2 matches `evaluate_variant`’s val estimate for the same supervised fit.
- No max-weight or long-only constraints in the optimizer call.
- High/low vol counts come from true labels {3,4} vs {0,1,2}.
