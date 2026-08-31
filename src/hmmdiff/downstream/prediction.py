# downstream_prediction.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def make_features_next_day_return(returns, window=5):
    """For each t, use returns[t-window..t-1] to predict returns[t]."""
    features, targets = [], []
    for i in range(window, len(returns)):
        features.append(returns[i - window : i])
        targets.append(returns[i])
    return np.array(features), np.array(targets)


def make_features_next_day_volatility(returns, window=5):
    """Use past returns to predict abs(return[t]) as a one-day vol proxy."""
    features, targets = [], []
    for i in range(window, len(returns)):
        features.append(returns[i - window : i])
        targets.append(abs(returns[i]))
    return np.array(features), np.array(targets)


def train_model_and_evaluate(real_data, synthetic_data=None, window=5, target="return"):
    """Elementary next-day return or volatility forecast with linear regression."""
    split_idx = int(0.8 * len(real_data))
    real_train = real_data[:split_idx]
    real_test = real_data[split_idx:]

    if target == "return":
        xr_train, yr_train = make_features_next_day_return(real_train, window)
        xr_test, yr_test = make_features_next_day_return(real_test, window)
    else:
        xr_train, yr_train = make_features_next_day_volatility(real_train, window)
        xr_test, yr_test = make_features_next_day_volatility(real_test, window)

    model_real = LinearRegression()
    model_real.fit(xr_train, yr_train)
    preds_real = model_real.predict(xr_test)
    mse_real = mean_squared_error(yr_test, preds_real)
    r2_real = r2_score(yr_test, preds_real)

    print(f"\n--- {target.capitalize()} Prediction: Train on Real Only ---")
    print(f"Test MSE: {mse_real:.6f},  R^2: {r2_real:.3f}")

    if synthetic_data is not None and len(synthetic_data) >= window + 1:
        if target == "return":
            xs_train, ys_train = make_features_next_day_return(synthetic_data, window)
        else:
            xs_train, ys_train = make_features_next_day_volatility(synthetic_data, window)

        model_synth = LinearRegression()
        model_synth.fit(xs_train, ys_train)
        preds_synth = model_synth.predict(xr_test)
        print(f"--- {target.capitalize()} Prediction: Train on Synthetic Only ---")
        print(f"Test MSE: {mean_squared_error(yr_test, preds_synth):.6f},  R^2: {r2_score(yr_test, preds_synth):.3f}")

        xc_train = np.concatenate([xr_train, xs_train], axis=0)
        yc_train = np.concatenate([yr_train, ys_train], axis=0)
        model_comb = LinearRegression()
        model_comb.fit(xc_train, yc_train)
        preds_comb = model_comb.predict(xr_test)
        print(f"--- {target.capitalize()} Prediction: Train on Real+Synthetic ---")
        print(f"Test MSE: {mean_squared_error(yr_test, preds_comb):.6f},  R^2: {r2_score(yr_test, preds_comb):.3f}")


VOL_FEATURE_COLS = ("Volatility", "MA_21", "RSI", "MACD")
VOL_TARGET_COL = "Future_Volatility"


def build_advanced_vol_features(df, price_col, intraday=False, horizon=21):
    """Build vol/TA features and a forward volatility target.

    Features are dated at observation time ``t``. ``Future_Volatility`` is the
    21-day realized vol ``horizon`` trading days later; ``target_date`` is that
    later calendar date.
    """
    import ta

    frame = df.copy().sort_index()
    frame["Returns"] = frame[price_col].pct_change()
    if not intraday:
        window_vol = 21
        frame["Volatility"] = frame["Returns"].rolling(window_vol).std() * np.sqrt(252)
    else:
        window_vol = 60
        frame["Volatility"] = frame["Returns"].rolling(window_vol).std() * np.sqrt(390)
    frame.dropna(subset=["Returns", "Volatility"], inplace=True)

    frame["MA_21"] = frame[price_col].rolling(window=21).mean()
    frame["RSI"] = ta.momentum.RSIIndicator(frame[price_col], window=14).rsi()
    frame["MACD"] = ta.trend.MACD(frame[price_col]).macd_diff()
    target_dates = pd.Series(frame.index, index=frame.index).shift(-int(horizon))
    frame[VOL_TARGET_COL] = frame["Volatility"].shift(-int(horizon))
    frame["target_date"] = target_dates
    frame.dropna(inplace=True)
    return frame


_build_advanced_vol_features = build_advanced_vol_features


def train_model_and_evaluate_advanced_volatility(
    df_real,
    df_synth=None,
    is_intraday=False,
    price_col="close",
    horizon=21,
):
    """Random-forest forward volatility forecast; three train scenarios."""
    real_feat = _build_advanced_vol_features(
        df_real, price_col=price_col, intraday=is_intraday, horizon=horizon
    )
    feature_cols = list(VOL_FEATURE_COLS)
    x_real = real_feat[feature_cols].values
    y_real = real_feat[VOL_TARGET_COL].values

    split_idx = int(0.8 * len(x_real))
    xr_train, yr_train = x_real[:split_idx], y_real[:split_idx]
    xr_test, yr_test = x_real[split_idx:], y_real[split_idx:]

    def train_and_report_scenario(xtrain, ytrain, xtest, ytest, scenario_label):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        mse = mean_squared_error(ytest, preds)
        r2 = r2_score(ytest, preds)
        print(f"\n--- ADV Vol: {scenario_label} ---")
        print(f"Test MSE: {mse:.6f},   R^2: {r2:.3f}")

        plt.figure(figsize=(8, 5))
        plt.plot(ytest, label="Actual Future Vol")
        plt.plot(preds, label=f"Predicted Vol ({scenario_label})")
        title_h = f"{horizon}-min" if is_intraday else f"{horizon}-day"
        plt.title(f"Daily Vol Forecast - {scenario_label} (horizon={title_h})")
        plt.legend()
        plt.show()
        return model, preds

    train_and_report_scenario(xr_train, yr_train, xr_test, yr_test, "Train on Real Only")

    if df_synth is not None:
        synth_feat = _build_advanced_vol_features(
            df_synth, price_col=price_col, intraday=is_intraday, horizon=horizon
        )
        x_synth = synth_feat[feature_cols].values
        y_synth = synth_feat[VOL_TARGET_COL].values
        train_and_report_scenario(x_synth, y_synth, xr_test, yr_test, "Train on Synthetic Only")
        xc_train = np.concatenate([xr_train, x_synth], axis=0)
        yc_train = np.concatenate([yr_train, y_synth], axis=0)
        train_and_report_scenario(xc_train, yc_train, xr_test, yr_test, "Train on Real+Synthetic")


def train_model_and_evaluate_advanced_volatility_mixture(
    df_real,
    df_synth=None,
    is_intraday=False,
    price_col="close",
    horizon=21,
    show_each_pred_plot=False,
    mix_grid=None,
    random_state=42,
    plot_summary=True,
    synth_source: str | None = None,
):
    """Sweep synthetic share in training; evaluate on the same real test set."""
    feat_real = _build_advanced_vol_features(
        df_real, price_col=price_col, intraday=is_intraday, horizon=horizon
    )
    feature_cols = list(VOL_FEATURE_COLS)
    x_real_all = feat_real[feature_cols].values
    y_real_all = feat_real[VOL_TARGET_COL].values

    split_idx = int(0.8 * len(x_real_all))
    xr_train, yr_train = x_real_all[:split_idx], y_real_all[:split_idx]
    xr_test, yr_test = x_real_all[split_idx:], y_real_all[split_idx:]

    if df_synth is not None:
        feat_synth = _build_advanced_vol_features(
            df_synth, price_col=price_col, intraday=is_intraday, horizon=horizon
        )
        x_synth_all = feat_synth[feature_cols].values
        y_synth_all = feat_synth[VOL_TARGET_COL].values
    else:
        x_synth_all = np.empty((0, len(feature_cols)))
        y_synth_all = np.empty(0)

    rng = np.random.default_rng(seed=random_state)
    if mix_grid is None:
        mix_grid = list(range(0, 101, 10))

    metrics = []
    n_common = min(len(xr_train), len(x_synth_all))
    x_real_subset = xr_train[:n_common]
    y_real_subset = yr_train[:n_common]
    x_synth_subset = x_synth_all[:n_common]
    y_synth_subset = y_synth_all[:n_common]

    for mix_pct in mix_grid:
        if mix_pct == 0:
            x_train, y_train = x_real_subset.copy(), y_real_subset.copy()
            label = "100 % Real"
        elif mix_pct == 100:
            x_train, y_train = x_synth_subset.copy(), y_synth_subset.copy()
            label = "100 % Synthetic"
        else:
            mask = rng.random(n_common) < (mix_pct / 100.0)
            x_train = np.where(mask[:, None], x_synth_subset, x_real_subset)
            y_train = np.where(mask, y_synth_subset, y_real_subset)
            actual_synth_pct = mask.sum() / n_common * 100
            label = f"{actual_synth_pct:.0f}% Synth + {100 - actual_synth_pct:.0f}% Real"

        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(xr_test)
        mse = mean_squared_error(yr_test, y_pred)
        r2 = r2_score(yr_test, y_pred)
        metrics.append((mix_pct, mse, r2))
        print(f"{label:>25s} | Test MSE = {mse:.6f} | R2 = {r2:+.3f}")

        if show_each_pred_plot:
            plt.figure(figsize=(7, 4))
            plt.plot(yr_test, label="Actual")
            plt.plot(y_pred, label="Predicted")
            title_h = f"{horizon}-min" if is_intraday else f"{horizon}-day"
            plt.title(f"Future Vol ({title_h}) – {label}")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()

    metrics_df = pd.DataFrame(metrics, columns=["synthetic_pct", "mse", "r2"])

    if plot_summary:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_mse = "tab:red"
        ax1.set_xlabel("% Synthetic in Training Set")
        ax1.set_ylabel("Test MSE", color=color_mse)
        ax1.plot(
            metrics_df["synthetic_pct"],
            metrics_df["mse"],
            marker="o",
            color=color_mse,
            label="Test MSE",
        )
        ax1.tick_params(axis="y", labelcolor=color_mse)

        ax2 = ax1.twinx()
        color_r2 = "tab:blue"
        ax2.set_ylabel("Test R²", color=color_r2)
        ax2.plot(
            metrics_df["synthetic_pct"],
            metrics_df["r2"],
            marker="s",
            linestyle="--",
            color=color_r2,
            label="Test R²",
        )
        ax2.tick_params(axis="y", labelcolor=color_r2)

        title_h = f"{horizon}-min" if is_intraday else f"{horizon}-day"
        title = f"Effect of Synthetic Share on Volatility Forecast ({title_h})"
        if synth_source:
            title = f"{title}\n[{synth_source}]"
        plt.title(title)
        fig.tight_layout()
        plt.show()

    return metrics_df


VOL_LAG_FEATURE_COLS = ("RV_21", "RV_21_lag5", "RV_21_lag21", "AbsRet_21")
DEFAULT_ADD_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def qlike_vol(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """QLIKE on variance (Patton): ``y/h - log(y/h) - 1`` with ``y, h`` = squared vol."""
    y = np.square(np.asarray(y_true, dtype=float))
    h = np.square(np.clip(np.asarray(y_pred, dtype=float), eps, None))
    y = np.clip(y, eps, None)
    return float(np.mean(y / h - np.log(y / h) - 1.0))


def build_vol_lag_features(df, price_col, horizon=21, rv_window=21):
    """Realized-vol lags only. No price-level MA / RSI / MACD.

    ``RV_21`` is the ``rv_window``-day annualized std ending at t. The target is that
    same series ``horizon`` trading days later. ``horizon == rv_window`` is the
    non-overlapping next-window RV.
    """
    frame = df.copy().sort_index()
    frame["Returns"] = frame[price_col].pct_change()
    ann = np.sqrt(252.0)
    rv = frame["Returns"].rolling(int(rv_window)).std() * ann
    frame["RV_21"] = rv
    frame["RV_21_lag5"] = rv.shift(5)
    frame["RV_21_lag21"] = rv.shift(int(rv_window))
    frame["AbsRet_21"] = frame["Returns"].abs().rolling(int(rv_window)).mean()
    target_dates = pd.Series(frame.index, index=frame.index).shift(-int(horizon))
    frame[VOL_TARGET_COL] = rv.shift(-int(horizon))
    frame["target_date"] = target_dates
    frame.dropna(inplace=True)
    return frame


def _synth_feature_rows(feat_synth: pd.DataFrame, feat_real_train_index) -> pd.DataFrame:
    """Date-aligned synth: train-period rows only. Unaligned pools: all rows."""
    if len(feat_synth) == 0:
        return feat_synth
    if isinstance(feat_synth.index, pd.DatetimeIndex) and isinstance(
        feat_real_train_index, pd.DatetimeIndex
    ):
        common = feat_synth.index.intersection(feat_real_train_index)
        if len(common) >= 8:
            return feat_synth.loc[common]
    return feat_synth


def _sample_xy(x, y, n, rng):
    if n <= 0 or len(x) == 0:
        return x[:0], y[:0]
    replace = n > len(x)
    idx = rng.choice(len(x), size=int(n), replace=replace)
    return x[idx], y[idx]


def train_vol_augmentation(
    df_real,
    df_synth=None,
    price_col="close",
    horizon=21,
    add_grid=None,
    n_seeds=5,
    random_state=42,
    n_estimators=100,
    synth_source: str | None = None,
    plot_summary=True,
):
    """Keep all real train rows; add a random synthetic subsample (multiples of n_train).

    Also reports persistence (``RV_21``) and HAR (linear vol lags) on real train only.
    """
    feat_real = build_vol_lag_features(df_real, price_col=price_col, horizon=horizon)
    feature_cols = list(VOL_LAG_FEATURE_COLS)
    split_idx = int(0.8 * len(feat_real))
    real_train = feat_real.iloc[:split_idx]
    real_test = feat_real.iloc[split_idx:]
    xr_train = real_train[feature_cols].to_numpy(dtype=float)
    yr_train = real_train[VOL_TARGET_COL].to_numpy(dtype=float)
    xr_test = real_test[feature_cols].to_numpy(dtype=float)
    yr_test = real_test[VOL_TARGET_COL].to_numpy(dtype=float)

    persist_hat = xr_test[:, feature_cols.index("RV_21")]
    har = LinearRegression()
    har.fit(xr_train, yr_train)
    har_hat = har.predict(xr_test)

    def _row(model, mse, r2, qlike, add_mult, seed):
        return {
            "model": model,
            "add_mult": float(add_mult),
            "seed": int(seed),
            "mse": float(mse),
            "r2": float(r2),
            "qlike": float(qlike),
            "n_real_train": int(len(xr_train)),
            "n_synth_added": 0,
        }

    rows = [
        _row(
            "persist",
            mean_squared_error(yr_test, persist_hat),
            r2_score(yr_test, persist_hat),
            qlike_vol(yr_test, persist_hat),
            0.0,
            0,
        ),
        _row(
            "har",
            mean_squared_error(yr_test, har_hat),
            r2_score(yr_test, har_hat),
            qlike_vol(yr_test, har_hat),
            0.0,
            0,
        ),
    ]

    xs_pool = np.empty((0, len(feature_cols)))
    ys_pool = np.empty(0)
    if df_synth is not None:
        feat_synth = build_vol_lag_features(df_synth, price_col=price_col, horizon=horizon)
        feat_synth = _synth_feature_rows(feat_synth, real_train.index)
        xs_pool = feat_synth[feature_cols].to_numpy(dtype=float)
        ys_pool = feat_synth[VOL_TARGET_COL].to_numpy(dtype=float)

    if add_grid is None:
        add_grid = DEFAULT_ADD_GRID

    print(
        f"real train {len(xr_train)}  test {len(xr_test)}  synth pool {len(xs_pool)}  "
        f"horizon {horizon}"
    )
    print(
        f"{'persist':>12s} | Test MSE = {rows[0]['mse']:.6f} | R2 = {rows[0]['r2']:+.3f} | "
        f"QLIKE = {rows[0]['qlike']:.4f}"
    )
    print(
        f"{'HAR':>12s} | Test MSE = {rows[1]['mse']:.6f} | R2 = {rows[1]['r2']:+.3f} | "
        f"QLIKE = {rows[1]['qlike']:.4f}"
    )

    for seed in range(int(n_seeds)):
        rng = np.random.default_rng(int(random_state) + seed)
        rf_state = int(random_state) + seed
        for add_mult in add_grid:
            n_add = int(round(float(add_mult) * len(xr_train)))
            xs, ys = _sample_xy(xs_pool, ys_pool, n_add, rng)
            if n_add > 0 and len(xs) == 0:
                raise ValueError("Requested synthetic rows but the synth pool is empty.")
            x_train = np.concatenate([xr_train, xs], axis=0) if len(xs) else xr_train
            y_train = np.concatenate([yr_train, ys], axis=0) if len(ys) else yr_train
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=rf_state)
            model.fit(x_train, y_train)
            y_pred = model.predict(xr_test)
            mse = mean_squared_error(yr_test, y_pred)
            r2 = r2_score(yr_test, y_pred)
            qlike = qlike_vol(yr_test, y_pred)
            rec = _row("rf", mse, r2, qlike, add_mult, seed)
            rec["n_synth_added"] = int(len(xs))
            rows.append(rec)
            tag = synth_source or "synth"
            print(
                f"seed {seed}  +{float(add_mult):.2f}x {tag:12s} | "
                f"n+={len(xs):4d} | Test MSE = {mse:.6f} | R2 = {r2:+.3f} | QLIKE = {qlike:.4f}"
            )

    metrics_df = pd.DataFrame(rows)
    if plot_summary:
        rf = metrics_df.loc[metrics_df["model"] == "rf"]
        summary = rf.groupby("add_mult", as_index=False).agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mse_mean=("mse", "mean"),
        )
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.errorbar(
            summary["add_mult"],
            summary["r2_mean"],
            yerr=summary["r2_std"].fillna(0.0),
            marker="s",
            color="tab:blue",
            label="RF test R² (mean ± sd)",
        )
        ax1.axhline(rows[0]["r2"], color="gray", ls=":", label="persist")
        ax1.axhline(rows[1]["r2"], color="tab:green", ls="--", label="HAR")
        ax1.set_xlabel("Synthetic rows added (× real train n)")
        ax1.set_ylabel("Test R²")
        title_h = f"{horizon}-day"
        title = f"Add synthetic data, keep all real train ({title_h})"
        if synth_source:
            title = f"{title}\n[{synth_source}]"
        ax1.set_title(title)
        ax1.legend(frameon=False)
        fig.tight_layout()
        plt.show()
    return metrics_df
