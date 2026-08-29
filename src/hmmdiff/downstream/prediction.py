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
        print(f"{label:>25s} | Test MSE = {mse:.6f} | R² = {r2:+.3f}")

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
