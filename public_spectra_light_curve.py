#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:36:58 2025

@author: Harry Nsubuga
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from pathlib import Path

import sncosmo
from sncosmo import Model, fit_lc, registry

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
        self.time_bins = np.arange(6, 44)

        pk = self._get_light_curve_dic()
        self.time_bins = np.array(pk["time_bin_center"])

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
  
if __name__ == "__main__":
    # --- Configuration ---
    system_number   = 1
    
    '''
    SN models:
     - “me” for the merger model
     - “n1” for the N100 model 
     - “su” for the sub-Ch model
     - “ww” for the W7 model
    '''
    supernova_model = "n1"
    input_data_path = Path("./data_release_holismokes7")
    n_sim           = 10000
    micro_config    = 9999
    #  six LSST filters u, g,r, i,z, and y, as well as the two infrared bands J and H
    filt            = "g" 
    # True macro-delay for (1 → 2)
    dt_true = 32.3 * u.day

    # Load two images
    k1, g1, s1, z1_src, z1_lens = f_get_system(system_number, 1)
    mlc1 = Mlcs(supernova_model, n_sim, k1, g1, s1,
                z1_src, z1_lens, input_data_path)
    k2, g2, s2, z2_src, z2_lens = f_get_system(system_number, 2)
    mlc2 = Mlcs(supernova_model, n_sim, k2, g2, s2,
                z2_src, z2_lens, input_data_path)

    pk = mlc1._get_light_curve_dic()
    true_bins = np.array(pk["time_bin_center"])
    mlc1.time_bins = mlc2.time_bins = true_bins

    t1, m1 = mlc1.load_microlensed_lightcurve(filt, micro_config)
    t2, m2 = mlc2.load_microlensed_lightcurve(filt, micro_config)
    t2 = t2 + dt_true

    # Convert mags to flux
    def mag_to_flux(mag, zp=25.0):
        return 10**(-0.4*(np.array(mag) - zp))

    flux1 = mag_to_flux(m1)
    flux2 = mag_to_flux(m2)
    err1  = 0.05 * flux1
    err2  = 0.05 * flux2

    # Build astropy Tables
    data1 = Table({
        'time':    t1.value,
        'band':    ['lsstg'] * len(t1),
        'flux':    flux1,
        'fluxerr': err1,
        'zp':      [25.0] * len(t1),
        'zpsys':   ['ab'] * len(t1),
    })
    data2 = Table({
        'time':    t2.value,
        'band':    ['lsstg'] * len(t2),
        'flux':    flux2,
        'fluxerr': err2,
        'zp':      [25.0] * len(t2),
        'zpsys':   ['ab'] * len(t2),
    })
        # --- Plot Macro vs Micro-lensed Light Curves ---
    plt.figure(figsize=(8,5))

    # Load macro versions
    t1_macro, m1_macro = mlc1.load_macrolensed_lightcurve(filt)
    t2_macro, m2_macro = mlc2.load_macrolensed_lightcurve(filt)

    plt.plot(t1.value, m1, 'r-', label=f"Microlensed (Image 1, config {micro_config})")
    plt.plot(t1_macro.value, m1_macro, 'r--', label="Macrolensed (Image 1)")

    plt.xlabel(f"Time after explosion [{t1.unit}] (observer frame)")
    plt.ylabel("Magnitude")
    plt.title("Macro vs Microlensed Light Curves")
    plt.legend()
    plt.gca().invert_yaxis()  
    plt.tight_layout()

    # --- Nugent fit with uncertainty ---
    nu1 = Model(source='nugent-sn1a'); nu1.set(z=z1_src)
    nu2 = Model(source='nugent-sn1a'); nu2.set(z=z1_src)

    t0_c1 = np.sum(t1.value * flux1) / np.sum(flux1)
    t0_c2 = np.sum(t2.value * flux2) / np.sum(flux2)

    # Seed t0 from the observed magnitude maximum (brightest point)
    t0_p1 = t1.value[np.argmax(flux1)]
    t0_p2 = t2.value[np.argmax(flux2)]
    
    # choose centroid if within data range, else fallback
    t0_guess1 = t0_c1 if t1.value.min()<=t0_c1<=t1.value.max() else t0_p1
    t0_guess2 = t0_c2 if t2.value.min()<=t0_c2<=t2.value.max() else t0_p2

    # Initial amplitude estimate from peak flux
    amp1 = flux1.max() / nu1.bandflux('lsstg', [t0_guess1], zp=25.0, zpsys='ab')[0]
    amp2 = flux2.max() / nu2.bandflux('lsstg', [t0_guess2], zp=25.0, zpsys='ab')[0]

    nu1.set(t0=t0_guess1, amplitude=amp1)
    nu2.set(t0=t0_guess2, amplitude=amp2)

    res1_n, fit1_n = fit_lc(data1, nu1, ['t0', 'amplitude'])
    res2_n, fit2_n = fit_lc(data2, nu2, ['t0', 'amplitude'])

    # Compute time-delay and uncertainty
    t0_1 = res1_n.parameters[res1_n.param_names.index('t0')]
    t0_2 = res2_n.parameters[res2_n.param_names.index('t0')]
    sigma1 = res1_n.errors['t0']
    sigma2 = res2_n.errors['t0']
    delta_t = t0_2 - t0_1
    sigma_dt = np.sqrt(sigma1**2 + sigma2**2)

    print(f"Nugent Δt = {delta_t:.2f} ± {sigma_dt:.2f} days")

    # Plot Nugent fits for both images
    sncosmo.plot_lc(data1, model=fit1_n, label='Image 1 (Nugent)', plot_phase=False)
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')
    plt.legend()

    sncosmo.plot_lc(data2, model=fit2_n, label='Image 2 (Nugent)', plot_phase=False)
    plt.xlabel('Time [days]')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()
