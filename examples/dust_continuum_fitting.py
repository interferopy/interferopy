import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.constants import c

from interferopy.tools import dust_lum, dust_sobs, sigfig


def dust_cont_fit():
    """
    Fit a modified black body dust continuum emission to observed flux density data points.
    File for input flux densities, redshift, and exact fitting method are currently hardcoded.
    Uses dust_sobs, which takes cmb heating and contrast into account by default.

    :return:
    """

    # Points to fit, inputs need so to be in SI units
    t = Table.read("./data/Pisco_continuum_fluxes.txt", format="ascii.commented_header")
    freqs = t["freq_ghz"] * 1e9  # to Hz
    fluxes = t["flux_jy"] * 1e-26  # to W/Hz/m2
    fluxes_err = t["flux_err_jy"] * 1e-26  # to W/Hz/m2

    # Parameters for the black body emission
    # These are either fixed, or used as intial guess for fitting if chosen to be free parameters
    z = 7.5413  # redshift
    dust_mass = 1e37  # in kilograms
    dust_temp = 47  # T_dust in Kelvins
    dust_beta = 1.9  # modified black body exponent

    dust_mass_err = 0
    dust_temp_err = 0
    dust_beta_err = 0

    # several fitting scenarios, choose one
    # lambdas are used to set non-free parameters, because curve_fit wants only the variable and free params
    # Parameters are degenerate, so be careful with error interpretation

    # Fit Mdust only - this could be used if you only have a single point, for example
    if 0:
        popt, pcov = curve_fit(lambda freqs, dust_mass:
                               dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass), sigma=fluxes_err, absolute_sigma=True)
        dust_mass = popt[0]
        dust_mass_err = np.diagonal(pcov)[0]

    # Fit Mdust and T - to constrain the temperature, the black body peak needs to be sampled
    if 0:
        popt, pcov = curve_fit(lambda freqs, dust_mass, dust_temp:
                               dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass, dust_temp), sigma=fluxes_err, absolute_sigma=True)
        dust_mass, dust_temp = popt
        dust_mass_err, dust_temp_err = np.sqrt(np.diagonal(pcov))

    # Fit Mdust and beta - the best option on the Rayleigh-Jeans tail
    if 1:
        popt, pcov = curve_fit(lambda freqs, dust_mass, dust_beta:
                               dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass, dust_beta), sigma=fluxes_err, absolute_sigma=True)
        dust_mass, dust_beta = popt
        dust_mass_err, dust_beta_err = np.sqrt(np.diagonal(pcov))

    # Fit Mdust and T and beta - not recommended due to degeneracy
    if 0:
        popt, pcov = curve_fit(lambda freqs, dust_mass, dust_temp, dust_beta:
                               dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass, dust_temp, dust_beta), sigma=fluxes_err,
                               absolute_sigma=True)
        dust_mass, dust_temp, dust_beta = popt
        dust_mass_err, dust_temp_err, dust_beta_err = np.sqrt(np.diagonal(pcov))

    print("dust_mass (10^8 Msol) = ", sigfig(dust_mass * u.kg.to(u.solMass) * 1e-8, 3),
          " +- ", sigfig(dust_mass_err * u.kg.to(u.solMass) * 1e-8, 1))
    print("dust_temp (K) = ", sigfig(dust_temp, 3), " +- ", sigfig(dust_temp_err, 1))
    print("dust_beta = ", sigfig(dust_beta, 3), " +- ", sigfig(dust_beta_err, 1))

    return dust_mass, dust_temp, dust_beta


def dust_cont_integrate(dust_mass, dust_temp, dust_beta):
    """
    Integrate over the IR spectral energy distribution. Calculate SFR based on Kennicut relation.
    Prints output to console.

    :param dust_mass: in kg
    :param dust_temp: in K
    :param dust_beta: dimensionless
    :return:
    """

    # Total IR is 8 - 1000 microns
    lum_tir = integrate.quad(lambda x: dust_lum(x, dust_mass, dust_temp, dust_beta), c / (1000e-6), c / (8e-6))
    print("Ltir (10^12 Lsol) = ", sigfig(lum_tir[0] * u.W.to(u.solLum) * 1e-12, 3))

    # Far IR is 42.5 - 122.5 microns
    lum_fir = integrate.quad(lambda x: dust_lum(x, dust_mass, dust_temp, dust_beta), c / (122.5e-6), c / (42.5e-6))
    print("Lfir (10^12 Lsol) =", sigfig(lum_fir[0] * u.W.to(u.solLum) * 1e-12, 3))

    # Kennicutt+98 relation scaled to Chabrier IMF
    print("SFR_Kennicutt98 (Msol/yr)", sigfig(lum_tir[0] * u.W.to(u.solLum) * 1e-10, 3))
    # print("SFR", Ltir[0]*4.5e-37/1.7) # Salpeter to Chabrier is a facotr of 1.7

    # Kennicutt+12 relation scaled to Chabrier IMF
    print("SFR_Kennicutt12 (Msol/yr)", sigfig(10 ** (np.log10(lum_tir[0] * u.W.to(u.erg / u.s)) - 43.41) / 1.7, 3))


def dust_cont_plot(z, dust_mass, dust_temp, dust_beta):
    """
    Plot the data points and the fitted dust continuum model.
    Input points file is hardcoded.

    :param z: redshift of the source
    :param dust_mass: in kg
    :param dust_temp: in K
    :param dust_beta: dimensionless
    :return:
    """

    # Points to plot
    t = Table.read("./data/Pisco_continuum_fluxes.txt", format="ascii.commented_header")
    freqs = t["freq_ghz"]
    fluxes = t["flux_jy"] * 1e3  # plot mJy
    fluxes_err = t["flux_err_jy"] * 1e3  # plot mJy

    fig, ax = plt.subplots(figsize=(4.8, 3))

    ax.errorbar(freqs, fluxes, marker="o", color="black", linestyle="", markerfacecolor='skyblue', ms=6,
                label='Measurements')
    ax.fill_between(freqs, fluxes - fluxes_err, fluxes + fluxes_err, color="skyblue", alpha=0.5)  # , label=r"aper err"

    # plot fitted model
    xxx = np.linspace(np.min(freqs) / 5, np.max(freqs) * 5, 1000)  # plot model over this range (GHz)
    ax.plot(xxx, dust_sobs(xxx * 1e9, z, dust_mass, dust_temp, dust_beta) * 1e26 * 1e3,  # x in GHz, y in mJy
            color="black", linestyle="-", lw=0.75, label="Fitted black body")

    # fill in the FIR region
    xxx = np.linspace(c / (122.5e-6) * 1e-9 / (1 + z), c / (42.5e-6) * 1e-9 / (1 + z), 1000)
    ax.fill_between(xxx, 0, dust_sobs(xxx * 1e9, z, dust_mass, dust_temp, dust_beta) * 1e26 * 1e3, color="orange",
                    alpha=0.2, label=r"FIR: rest-frame 42.5 - 122.5 $\mu$m")

    # axes details
    ax.set_xlabel(r"$\nu_{obs}$ (GHz)")
    ax.set_ylabel(r"$S_\nu$ (mJy)")  # observed flux densities
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1.4)
    ax.tick_params(direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # add a second axis to display wavelengths in mm
    # wavelength scales as 1/freq, so I manually put ticks to be sure they are in the right spot
    label_mm = np.array([0.2, 0.3, 0.4, 0.5, 1, 2, 5])  # plot these mm ticks
    label_mm_corresponding_ghz = c / (label_mm * 1e-3) * 1e-9  # get at which frequencies do these ticks occur
    label_mm_format = [x if x < 1 else int(x) for x in label_mm]  # do not write decimal part for ticks > 1

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # match the range to the lower axis (make sure manual ticks are within this range)
    ax2.set_xticks(label_mm_corresponding_ghz)
    ax2.set_xticklabels(label_mm_format)  # override tick labels to mm
    ax2.set_xlabel(r"$\lambda_{obs}$ (mm)")
    ax2.tick_params(direction='in')

    plt.savefig("./plots/dust_continuum_fluxes.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("./thumbnails/dust_continuum_fluxes.png", bbox_inches="tight", dpi=72)
    plt.show()


def main():
    # redshift of pisco
    z = 7.5413

    dust_params = dust_cont_fit()  # = dust_mass, dust_temp, dust_beta
    dust_cont_integrate(*dust_params)
    dust_cont_plot(z, *dust_params)


if __name__ == "__main__":
    main()
