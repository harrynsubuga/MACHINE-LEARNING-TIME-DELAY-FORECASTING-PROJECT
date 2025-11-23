#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end Δt fit with a custom sncosmo source built from HoliSmokes macro SEDs.
- Uses flux (no magnitudes for fitting/plotting)
- Fits image 1 and image 2 separately, then Δt = t0_2 - t0_1
- Presentation-quality plot with axis starting at 0 and Δt annotation
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from pathlib import Path

import sncosmo
from sncosmo import Model, fit_lc, TimeSeriesSource

# ------------------------------- Fixed system metadata -------------------------------

def f_get_system(system_number, image_number):
    sn = system_number
    kappa = gamma = s = None
    source_redshift = lens_redshift = None

    if sn in (1, 2, 3, 4, 5):
        s = 0.600
        if image_number == 1:
            kappa, gamma = 0.250895, 0.274510
        elif image_number == 2:
            kappa, gamma = 0.825271, 0.814777
        if sn == 1:
            source_redshift, lens_redshift = 0.76, 0.252
        elif sn == 2:
            source_redshift, lens_redshift = 0.55, 0.252
        elif sn == 3:
            source_redshift, lens_redshift = 0.99, 0.252
        elif sn == 4:
            source_redshift, lens_redshift = 0.76, 0.16
        elif sn == 5:
            source_redshift, lens_redshift = 0.76, 0.48

    elif sn in (6, 7, 8):
        s = {6: 0.3, 7: 0.59, 8: 0.9}[sn]
        source_redshift, lens_redshift = 0.76, 0.252
        if image_number == 1:
            kappa, gamma = 0.250895, 0.274510
        elif image_number == 2:
            kappa, gamma = 0.825271, 0.814777

    elif sn == 9:
        s = 0.6
        source_redshift, lens_redshift = 0.76, 0.252
        if image_number == 1:
            kappa, gamma = 0.434950, 0.414743
        elif image_number == 2:
            kappa, gamma = 0.431058, 0.423635
        elif image_number == 3:
            kappa, gamma = 0.566524, 0.536502
        elif image_number == 4:
            kappa, gamma = 1.282808, 1.252791

    else:
        raise ValueError(f"Unknown system_number: {sn}")

    return kappa, gamma, s, source_redshift, lens_redshift


class Mlcs:
    def __init__(self, supernova_model, n_sim, kappa, gamma, s,
                 source_redshift, lens_redshift, input_data_path: Path):
        self.supernova_model = supernova_model
        self.n_sim = n_sim
        self.kappa = kappa
        self.gamma = gamma
        self.s = s
        self.source_redshift = source_redshift
        self.lens_redshift = lens_redshift
        self.input_data_path = input_data_path
        self.data_version = "IRreduced"
        self.time_bins = None  # set below with units

        pk = self._get_light_curve_dic()
        # Preserve units if present
        self.time_bins = pk["time_bin_center"]  # likely an astropy Quantity

    def _get_light_curve_dic(self):
        fname = (f"k{self.kappa:.6f}_g{self.gamma:.6f}_"
                 f"s{self.s:.3f}_redshift_source_{self.source_redshift:.3f}_"
                 f"lens{self.lens_redshift:.3f}_Nsim_{self.n_sim}.pickle")
        fullpath = self.input_data_path / "light_curves" / fname
        with open(fullpath, "rb") as handle:
            return pickle.load(handle, encoding="latin1")

    def load_microlensed_lightcurve(self, filt, micro_config):
        d = self._get_light_curve_dic()
        key = f"micro_light_curve_{self.supernova_model}{micro_config}{filt}"
        return d["time_bin_center"], d[key]

    def load_macrolensed_lightcurve(self, filt):
        d = self._get_light_curve_dic()
        key = f"macro_light_curve_{self.supernova_model}{filt}"
        return d["time_bin_center"], d[key]

    def _get_flux_dic(self, time_bin):
        dirname = (f"{self.supernova_model}/"
                   f"k{self.kappa:.6f}_g{self.gamma:.6f}_"
                   f"s{self.s:.3f}_redshift_source{self.source_redshift:.3f}_"
                   f"lens{self.lens_redshift:.3f}_Nsim_{self.n_sim}")
        spectra_dir = self.input_data_path / "spectra" / dirname
        meta_fname = (f"{self.supernova_model}_k{self.kappa:.6f}_g{self.gamma:.6f}_"
                      f"s{self.s:.3f}_redshift_source{self.source_redshift:.3f}_"
                      f"lens{self.lens_redshift:.3f}_Nsim_{self.n_sim}.pickle")
        with open(self.input_data_path / "LSNeIa_class" / meta_fname, "rb") as handle:
            SNmicro = pickle.load(handle, encoding="latin1")
        t = SNmicro.time_bin_center[time_bin].to(u.day).value
        picklename = f"time_{t:.2f}.pickle"
        with open(spectra_dir / picklename, "rb") as handle:
            return pickle.load(handle, encoding="latin1"), SNmicro.time_bin_center[time_bin]

    def load_microlensed_flux(self, micro_config, time_bin):
        d_flux, time = self._get_flux_dic(time_bin)
        return d_flux["lam_bin_center"], d_flux[f"micro_flux_{micro_config}"], time

    def load_macrolensed_flux(self, time_bin):
        d_flux, time = self._get_flux_dic(time_bin)
        return d_flux["lam_bin_center"], d_flux["macro_flux"], time


# ------------------------------- Helpers -------------------------------

def mag_to_flux(mag, zp=25.0):
    """AB mag -> flux in arbitrary linear units compatible with sncosmo bandflux amplitude scaling."""
    return 10.0 ** (-0.4 * (np.array(mag) - zp))

def _as_days(arr):
    """Return numpy array of days; accept astropy Quantity or plain ndarray/list."""
    if hasattr(arr, "to"):
        return arr.to(u.day).value
    return np.asarray(arr, dtype=float)

def build_holismokes_source_from_macro_seds(mlc, name="holi-su"):
    """
    Build a sncosmo TimeSeriesSource from MACRO (unmicrolensed) spectra on the observer frame grid.
    We set z=0 later so sncosmo does not apply any extra redshift/time-dilation.
    """
    phases = _as_days(mlc.time_bins)  # observer-frame days since explosion in the files
    wave = None
    flux_grid = []
    for ibin in range(len(phases)):
        wave_i, macro_flux_i, _ = mlc.load_macrolensed_flux(ibin)
        if wave is None:
            wave = np.array(wave_i)  # Angstrom (observer-frame)
        flux_grid.append(np.array(macro_flux_i))
    flux_grid = np.array(flux_grid)  # shape (Nphase, Nwave)

    _ = TimeSeriesSource(phase=phases, wave=wave, flux=flux_grid, name=name)  # registers by name
    return name

def common_overlap_mask(t1, t2):
    """Return masks that clip to the common time overlap region of two arrays."""
    tmin = max(t1.min(), t2.min())
    tmax = min(t1.max(), t2.max())
    m1 = (t1 >= tmin) & (t1 <= tmax)
    m2 = (t2 >= tmin) & (t2 <= tmax)
    return m1, m2

def smooth_series(y, window=7):
    """Light moving-average smoothing for presentation."""
    w = max(3, int(window) | 1)  # odd >=3
    if len(y) < w:
        return y
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode='valid')


# ------------------------------- Main -------------------------------

if __name__ == "__main__":
    # --- Configuration ---
    system_number   = 1
    supernova_model = "su"                   # "me", "n1", "su", or "ww"
    input_data_path = Path("./data_release_holismokes7")
    n_sim           = 10000
    micro_config    = 9999
    filt            = "z"                    # LSST filter key in your files (use 'z')
    dt_true         = 32.3 * u.day           # Injected macro delay (1 -> 2)

    # --- Instantiate both images first ---
    k1, g1, s1, z1_src, z1_lens = f_get_system(system_number, 1)
    mlc1 = Mlcs(supernova_model, n_sim, k1, g1, s1, z1_src, z1_lens, input_data_path)

    k2, g2, s2, z2_src, z2_lens = f_get_system(system_number, 2)
    mlc2 = Mlcs(supernova_model, n_sim, k2, g2, s2, z2_src, z2_lens, input_data_path)

    # --- Now sync their time grids (preserving units) ---
    pk = mlc1._get_light_curve_dic()
    true_bins = pk["time_bin_center"]  # keep as Quantity if provided
    mlc1.time_bins = mlc2.time_bins = true_bins

    # --- Load microlensed mags and apply injected macro delay to image 2 times ---
    t1_q, m1_mag = mlc1.load_microlensed_lightcurve(filt, micro_config)
    t2_q, m2_mag = mlc2.load_microlensed_lightcurve(filt, micro_config)

    t1 = _as_days(t1_q)
    t2 = _as_days(t2_q + dt_true)

    # --- Convert to flux and assign nominal 5% uncertainties ---
    flux1 = mag_to_flux(m1_mag)
    flux2 = mag_to_flux(m2_mag)
    err1  = 0.05 * flux1
    err2  = 0.05 * flux2

    # --- Build sncosmo photometry tables (z-band only here) ---
    data1 = Table({
        'time':    t1,
        'band':    ['lsstz'] * len(t1),
        'flux':    flux1,
        'fluxerr': err1,
        'zp':      [25.0] * len(t1),
        'zpsys':   ['ab'] * len(t1),
    })
    data2 = Table({
        'time':    t2,
        'band':    ['lsstz'] * len(t2),
        'flux':    flux2,
        'fluxerr': err2,
        'zp':      [25.0] * len(t2),
        'zpsys':   ['ab'] * len(t2),
    })

    # --- Build & register a custom source from the MACRO SED grid (observer frame) ---
    src_name = build_holismokes_source_from_macro_seds(mlc1, name=f"holi-{supernova_model}")

    m1 = sncosmo.Model(source=src_name)  # IMPORTANT: observer-frame source => set z=0
    m2 = sncosmo.Model(source=src_name)
    m1.set(z=0.0)
    m2.set(z=0.0)

    # --- Seed t0 near peak & amplitude from peak flux in the correct band (lsstz) ---
    t0_guess1 = t1[np.argmax(flux1)]
    t0_guess2 = t2[np.argmax(flux2)]
    m1.set(t0=t0_guess1)
    m2.set(t0=t0_guess2)

    # Avoid zero division; bandflux returns an array
    fmod1 = m1.bandflux('lsstz', [t0_guess1], zp=25.0, zpsys='ab')[0]
    fmod2 = m2.bandflux('lsstz', [t0_guess2], zp=25.0, zpsys='ab')[0]
    amp1_init = (flux1.max() / fmod1) if fmod1 != 0 else 1.0
    amp2_init = (flux2.max() / fmod2) if fmod2 != 0 else 1.0
    m1.set(amplitude=amp1_init)
    m2.set(amplitude=amp2_init)

    # --- Fit t0 and amplitude per image ---
    res1, fit1 = fit_lc(data1, m1, ['t0', 'amplitude'])
    res2, fit2 = fit_lc(data2, m2, ['t0', 'amplitude'])

    def get_param(results, name):
        idx = results.param_names.index(name)
        val = results.parameters[idx]
        err = results.errors.get(name, np.nan) if results.errors is not None else np.nan
        return val, err

    t0_1, sig1 = get_param(res1, 't0')
    t0_2, sig2 = get_param(res2, 't0')
    delta_t    = t0_2 - t0_1
    sigma_dt   = np.hypot(sig1 if np.isfinite(sig1) else 0.0,
                          sig2 if np.isfinite(sig2) else 0.0)

    print(f"Custom-template Δt = {delta_t:.2f} ± {sigma_dt:.2f} days (true: {(_as_days(dt_true))[()] if hasattr(dt_true,'unit') else dt_true:.2f} d)")

    # -------------------------- Presentation-quality plot --------------------------
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # Clip to common overlap for a fair visual comparison
    msk1, msk2 = common_overlap_mask(t1, t2)
    t1c, f1c, e1c = t1[msk1], flux1[msk1], err1[msk1]
    t2c, f2c, e2c = t2[msk2], flux2[msk2], err2[msk2]

    # Start x-axis at zero for plotting only (do not alter fit)
    t0_plot = min(t1c.min(), t2c.min())
    t1p = t1c - t0_plot
    t2p = t2c - t0_plot

    # Smoothed guides
    f1g = smooth_series(f1c, window=7)
    f2g = smooth_series(f2c, window=7)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(t1p, f1c, yerr=e1c, fmt='o', ms=3, lw=0.8, alpha=0.6, label='Image 1 (data)')
    ax.errorbar(t2p, f2c, yerr=e2c, fmt='s', ms=3, lw=0.8, alpha=0.6, label='Image 2 (data)')

    ax.plot(t1p, f1g, '-',  lw=1.2, alpha=0.9, label='Image 1 (smooth)')
    ax.plot(t2p, f2g, '--', lw=1.2, alpha=0.9, label='Image 2 (smooth)')

    ax.set_xlim(0, max(t1p.max(), t2p.max()))
    ax.set_xlabel("Time since first observation [days]")  # axis starts at 0
    ax.set_ylabel("Flux (arb. units)")
    ax.set_title("Microlensed LSST-z Light Curves (Image 1 vs Image 2)")
    ax.grid(True, which='major', alpha=0.15)
    ax.grid(True, which='minor', alpha=0.07)
    ax.minorticks_on()

    txt = r"$\Delta t$" + f" = {delta_t:.2f} ± {sigma_dt:.2f} d  (true {(_as_days(dt_true))[()] if hasattr(dt_true,'unit') else dt_true:.2f} d)"
    ax.text(0.98, 0.02, txt, ha='right', va='bottom', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='0.7', alpha=0.9))

    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()

    # -------------------------- (Optional) Overlay after aligning by fitted Δt --------------------------
    t2_aligned = t2c - delta_t
    t1p2 = t1c - t0_plot
    t2p2 = t2_aligned - t0_plot

    f2g_align = smooth_series(f2c, window=7)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.2))
    ax2.errorbar(t1p2, f1c, yerr=e1c, fmt='o', ms=3, lw=0.8, alpha=0.6, label='Image 1 (data)')
    ax2.errorbar(t2p2, f2c, yerr=e2c, fmt='s', ms=3, lw=0.8, alpha=0.6, label='Image 2 (shifted by −Δt)')
    ax2.plot(t1p2, f1g, '-', lw=1.2, alpha=0.9, label='Image 1 (smooth)')
    ax2.plot(t2p2, f2g_align, '--', lw=1.2, alpha=0.9, label='Image 2 (smooth, shifted)')

    ax2.set_xlim(0, max(t1p2.max(), t2p2.max()))
    ax2.set_xlabel("Time since first observation [days]")
    ax2.set_ylabel("Flux (arb. units)")
    ax2.set_title("Aligned by fitted Δt")
    ax2.grid(True, which='major', alpha=0.15)
    ax2.grid(True, which='minor', alpha=0.07)
    ax2.minorticks_on()
    ax2.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()
