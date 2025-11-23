"""
RFv4_bias_minimized.py

Machine-learning estimator for recovering macro time delays 
between strongly-lensed Type Ia Supernova images.

Key features of RFv4:
    • Oversampling of long delays (30–70 days) to reduce regression bias
    • One lag-aware feature: cross-correlation peak lag (in days)
    • Linear calibration on a held-out validation split
    • Robust evaluation across many microlensing configurations

Author: Harry Nsubuga
Placement: SEPnet Summer Placement @ ICG Portsmouth (2024)
Supervisor: Dr Ana Sainz

This script is linked on my SEPnet EXPO poster via QR code.
"""

# ============================================================
# ======================= IMPORTS =============================
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from scipy.interpolate import interp1d
from scipy.signal import correlate
from astropy import units as u

# H0LiSmokes dataset loader
from public_spectra_light_curve import f_get_system, Mlcs


# ============================================================
# ======================= HELPERS =============================
# ============================================================

def shift_light_curve_extrap(time_bins, mags, delta_t):
    """
    Shift a light curve by delta_t days onto the common time grid.

    Parameters
    ----------
    time_bins : array
        The interpolation grid.
    mags : array
        Original microlensed magnitudes.
    delta_t : float
        Time delay to apply (days).

    Returns
    -------
    array : shifted magnitudes aligned to time_bins.
    """
    f = interp1d(time_bins, mags, bounds_error=False, fill_value="extrapolate")
    return f(time_bins - delta_t)


def xcorr_lag_days(a, b, time_bins):
    """
    Compute cross-correlation lag (in days) between two light curves.

    Notes
    -----
    This provides a very rough delay estimator. It is NOT used as the
    final prediction, but included as an auxiliary ML feature.

    Returns
    -------
    float : lag in days corresponding to the correlation peak.
    """
    a0, b0 = a - np.nanmean(a), b - np.nanmean(b)
    c = correlate(a0, b0, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    dt = float(np.median(np.diff(time_bins)))
    return lags[np.argmax(c)] * dt


# ============================================================
# ============== TRAINING-SET CONSTRUCTION ===================
# ============================================================

def prepare_training(mlc1, mlc2, filt, n_samples=2000, seed=0):
    """
    Construct the training matrix X and target array y for RFv4.

    The training samples are built by:
        1. selecting two microlensed light-curve realizations,
        2. injecting a known delay (0–30d small OR 30–70d oversampled),
        3. adding small magnitude offsets (-0.4–0.4 mag),
        4. normalising curves,
        5. computing a lag-aware feature via cross-correlation,
        6. concatenating [LC1 | LC2 | lagFeature].

    Parameters
    ----------
    mlc1, mlc2 : Mlcs objects
        Two images of the same strongly-lensed SN Ia system.
    filt : str
        Photometric band ("g", "r", "i", "z").
    n_samples : int
        Size of the generated training set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : (n_samples, 2*n_bins + 1) array
        Input features.
    y : (n_samples,) array
        True delays used to generate each pair.
    """
    d1 = mlc1._get_light_curve_dic()
    d2 = mlc2._get_light_curve_dic()

    tb = mlc1.time_bins
    n_bins = len(tb)
    rng = np.random.RandomState(seed)

    # -------------------------
    # Oversample long delays
    # -------------------------
    n_big = n_samples // 2
    small = rng.uniform(0, 30, n_samples - n_big)
    big   = rng.uniform(30, 70, n_big)
    time_delays = np.concatenate([small, big])
    rng.shuffle(time_delays)

    X = np.empty((n_samples, n_bins * 2 + 1))
    mag_shift_range = (-0.4, 0.4)

    # Random indices for simulated microlensing realizations
    idx1 = rng.randint(0, mlc1.n_sim, n_samples)
    idx2 = rng.randint(0, mlc2.n_sim, n_samples)

    for i in range(n_samples):
        # --- Image 1 ---
        key1 = f"micro_light_curve_{mlc1.supernova_model}{idx1[i]}{filt}"
        m1 = np.array(d1[key1])
        m1 += rng.uniform(*mag_shift_range)
        n1 = m1 - np.min(m1)

        # --- Image 2 (shifted by known delay) ---
        key2 = f"micro_light_curve_{mlc2.supernova_model}{idx2[i]}{filt}"
        m2 = np.array(d2[key2])
        m2_shifted = shift_light_curve_extrap(tb, m2, time_delays[i])
        m2_shifted += rng.uniform(*mag_shift_range)
        n2 = m2_shifted - np.min(m2_shifted)

        # --- Lag-aware cross-correlation feature ---
        lag_feat = xcorr_lag_days(n1, n2, tb)

        X[i] = np.concatenate([n1, n2, [lag_feat]])

    return X, time_delays


# ============================================================
# ========================== MAIN ============================
# ============================================================

if __name__ == "__main__":

    # ------------------- CONFIGURATION ------------------- #
    system_number = 1
    supernova_model = "su"
    input_data_path = Path("./data_release_holismokes7")
    n_sim = 10000
    filt = "z"
    micro_config = 9999
    dt_true = 32.3 * u.day

    # ------------------- LOAD DATA ----------------------- #
    k1, g1, s1, z1_src, z1_lens = f_get_system(system_number, 1)
    mlc1 = Mlcs(supernova_model, n_sim, k1, g1, s1,
                z1_src, z1_lens, input_data_path)

    k2, g2, s2, z2_src, z2_lens = f_get_system(system_number, 2)
    mlc2 = Mlcs(supernova_model, n_sim, k2, g2, s2,
                z2_src, z2_lens, input_data_path)

    # ------------------- TRAINING ------------------------ #
    X, y = prepare_training(mlc1, mlc2, filt, n_samples=20000, seed=42)

    # Train/val split
    n = len(y)
    n_val = max(2000, n // 10)
    X_tr, y_tr = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    # Random Forest model
    rf = RandomForestRegressor(
        n_estimators=1200,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    # ------------------- CALIBRATION --------------------- #
    y_pred_val = rf.predict(X_val)
    a, b = np.polyfit(y_pred_val, y_val, 1)
    print(f"Calibration: y_true ≈ {a:.4f} * y_pred + {b:.4f}")

    # ------------------- SINGLE TEST --------------------- #
    tb = mlc1.time_bins
    t1, m1 = mlc1.load_microlensed_lightcurve(filt, micro_config)
    t2, m2 = mlc2.load_microlensed_lightcurve(filt, micro_config)

    m1b = interp1d(t1, m1, fill_value='extrapolate')(tb)
    m2b = interp1d(t2, m2, fill_value='extrapolate')(tb)
    m2_shifted = shift_light_curve_extrap(tb, m2b, float(dt_true.value))

    n1 = m1b - m1b.min()
    n2 = m2_shifted - m2_shifted.min()
    lag_feat = xcorr_lag_days(n1, n2, tb)

    X_test = np.hstack([n1, n2, [lag_feat]]).reshape(1, -1)

    dt_pred_raw = rf.predict(X_test)[0]
    dt_pred = a * dt_pred_raw + b

    per_tree = np.array([t.predict(X_test)[0] for t in rf.estimators_])
    sigma = per_tree.std()

    print(f"Predicted delay: {dt_pred:.3f} d | ±{sigma:.3f} d")
    print(f"True delay: {dt_true:.2f}")

    # ============================================================
    # ============== ROBUST EVALUATION (PLOT + METRICS) ==========
    # ============================================================

    rng = np.random.RandomState(0)
    micro_list = rng.choice(mlc1.n_sim, size=40, replace=False)
    true_delays = np.linspace(5, 60, 20)

    mean_pred, std_pred = [], []
    bias_mean, bias_std = [], []

    all_true, all_pred = [], []

    for dt in true_delays:
        preds = []
        for mc in micro_list:
            t1, m1 = mlc1.load_microlensed_lightcurve(filt, mc)
            t2, m2 = mlc2.load_microlensed_lightcurve(filt, mc)

            m1b = interp1d(t1, m1, fill_value='extrapolate')(tb)
            m2b = interp1d(t2, m2, fill_value='extrapolate')(tb)
            m2_shifted = shift_light_curve_extrap(tb, m2b, float(dt))

            n1 = m1b - m1b.min()
            n2 = m2_shifted - m2_shifted.min()
            lag_feat = xcorr_lag_days(n1, n2, tb)

            x = np.hstack([n1, n2, [lag_feat]]).reshape(1, -1)

            p_raw = rf.predict(x)[0]
            p = a * p_raw + b
            preds.append(p)

            all_true.append(dt)
            all_pred.append(p)

        preds = np.array(preds)
        mean_pred.append(np.mean(preds))
        std_pred.append(np.std(preds))

        bias = preds - dt
        bias_mean.append(np.mean(bias))
        bias_std.append(np.std(bias))

    # Convert to arrays
    mean_pred = np.array(mean_pred)
    std_pred = np.array(std_pred)
    bias_mean = np.array(bias_mean)
    bias_std = np.array(bias_std)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # ============================================================
    # ============== GLOBAL METRICS (POSTER TABLE) ===============
    # ============================================================

    mae = mean_absolute_error(all_true, all_pred)
    bias_global = np.mean(all_pred - all_true)
    r2 = r2_score(all_true, all_pred)

    # Mean model uncertainty over micro-configs
    mean_sigma = np.mean(std_pred)
    std_sigma  = np.std(std_pred)

    # Runtime per prediction
    start_t = time.time()
    _ = rf.predict(x)
    runtime_ms = (time.time() - start_t) * 1000.0

    print("\n--- RFv4 Performance Summary (Poster Table) ---")
    print(f"MAE (days): {mae:.2f}")
    print(f"Mean Bias (days): {bias_global:+.2f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Runtime per prediction (ms): {runtime_ms:.1f}")
    print(f"Model uncertainty σΔt (days): {mean_sigma:.2f} ± {std_sigma:.2f}")

    # ============================================================
    # ===================== RESULTS PLOTS ========================
    # ============================================================

    plt.figure()
    plt.errorbar(true_delays, mean_pred, yerr=std_pred,
                 fmt='o', capsize=3, label='Predicted ± std')
    plt.plot(true_delays, true_delays, '--', label='Ideal')
    plt.xlabel('True delay (days)')
    plt.ylabel('Predicted delay (days)')
    plt.title('RFv4 Delay Recovery (lag feature + calibration)')
    plt.legend()

    plt.figure()
    plt.errorbar(true_delays, bias_mean, yerr=bias_std,
                 fmt='o', capsize=3)
    plt.axhline(0, linestyle='--')
    plt.xlabel('True delay (days)')
    plt.ylabel('Bias (days)')
    plt.title('Bias vs True Delay (Mean ± Std)')

    plt.show()