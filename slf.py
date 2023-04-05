# Python 3.8.16
import os
import pickle
import datetime

# Numpy 1.24.2
import numpy as np
from numpy.random import default_rng
# matplotlib 3.7.1
import matplotlib.pyplot as plt
# astropy 5.2.1
from astropy import units as u
from astropy.coordinates import Distance
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
# scipy 1.9.3
from scipy.special import j1
from scipy import optimize
from scipy.stats import truncnorm
from scipy.ndimage import gaussian_filter as gf
# mpmath 1.3.0
from mpmath import gammainc

# Classes and functions

class Telescope:
    """
    Telescope parameters
    """
    parameters = {
        'dummy': {  # test parameters
            'sefd': 15 * u.Jy,
            'v0': 800 * u.MHz,
            'v1': 1600 * u.MHz,
            'D': 50 * u.m,
            'bands': [
                [800 * u.MHz, 1200 * u.MHz],
                [1200 * u.MHz, 1600 * u.MHz],
            ]
        },
        'chime': {  # from https://ui.adsabs.harvard.edu/abs/2022ApJS..261...29C/abstract
            # from https://ui.adsabs.harvard.edu/abs/2017ApJ...844..140C/abstract
            'sefd': (50 + 34 * (600/408)**-2.6) / 1.38 * u.Jy,
            'v0': 400 * u.MHz,
            'v1': 800 * u.MHz,            
            'Dx': 20 * u.m,
            'Dy': 0.87 * u.m,  # Emprirical value to have a beam of ~100 deg
            #'D': 8.4474307*u.m
        },
        'chord': {  # from https://ui.adsabs.harvard.edu/abs/2019clrp.2020...28V/abstract
            'sefd': 9 * u.Jy,
            'v0': 300 * u.MHz,
            'v1': 1500 * u.MHz,
            'D': 6 * u.m,
            'bands': [
                [300 * u.MHz, 700 * u.MHz],
                [700 * u.MHz, 1100 * u.MHz],
                [1100 * u.MHz, 1500 * u.MHz],
            ]
        },
        'dsa2000': {  # from https://ui.adsabs.harvard.edu/abs/2019BAAS...51g.255H/abstract
            'sefd': 2.5 * u.Jy,  # Jy
            'v0': 700 * u.MHz,
            'v1': 2000 * u.MHz,
            'D': 5 * u.m,
            'bands': [
                [700 * u.MHz, 1100 * u.MHz],
                [1100 * u.MHz, 1500 * u.MHz],
                [1500 * u.MHz, 2000 * u.MHz],
            ]
        },
    }
    
    def __init__(self, name='chord'):
        self.name = name
        self.parameters = Telescope.parameters
        
    def get_parameters(self, name = None):
        if name is None: name = self.name
        return self.parameters[name]
    
    def bandwidth(self, name = None):
        if name is None: name = self.name
        return self.parameters[name]['v1'] - self.parameters[name]['v0']
    
    def sefd(self, name = None):
        if name is None: name = self.name
        return self.parameters[name]['sefd']  
    
    def v0(self, name = None):
        if name is None: name = self.name
        return self.parameters[name]['v0']
    
    def v1(self, name = None):
        if name is None: name = self.name
        return self.parameters[name]['v1']
    
    def get_parameter(self, parameter, name = None):
        if name is None: name = self.name
        return self.parameters[name][parameter]  
    
    
def get_lensed_fraction(
    z
):
    """ Fraction of lensed galaxies as a function of redshift.
    
    """
    if frbs_evolve_with_luminosity:
        # Unlensed population of galaxies used in Collett 2018 simulations
        z_org = np.loadtxt('lsst_source_catalog.txt', usecols=2, delimiter=',')
        # Luminosity of unlensed population of galaxies used in Collett 2018 simulations
        M_org = np.loadtxt('lsst_source_catalog.txt', usecols=7, delimiter=',') * u.M_bol
        L_org = M_org.to(u.L_bol) * 10 * 360**2 / np.pi  # The population covers 0.1 square deg
        # Total luminosity of unlensed galaxy population per redshift step
        PL_org, _ = np.histogram(
            z_org, 
            bins=z, 
            weights=L_org
        )
        # Luminosity of lensed galaxies used from Collett 2018 simulations
        L_sim = lensed_galaxies_Mv.to(u.L_sun)
        # Total luminosity of lensed galaxy population per redshift step
        PL_sim, _ = np.histogram(
            lensed_galaxies_z, 
            bins=z, 
            weights=L_sim
        )
        # Fraction of luminosity of lensed galaxies per redshift step
        lensed_fraction = PL_sim / PL_org

    else:
        # Unlensed population of galaxies used in Collett 2018 simulations
        z_org = np.loadtxt('lsst_source_catalog.txt', usecols=2, delimiter=', ')
        Pz_org, _ = np.histogram(z_org, bins=z)
        Pz_org = Pz_org * 10 * 360**2 / np.pi  # The population covers 0.1 square deg
        # Lensed population of galaxies from Collett 2018 simulations
        Pz_sim, _ = np.histogram(lensed_galaxies_z, bins=z)
        # Fraction of lensed galaxies per redshift step
        lensed_fraction = Pz_sim / Pz_org
        
    # Smooth the output
    lensed_fraction = gf(lensed_fraction, 6)
    return lensed_fraction



def get_distribution_z(
    lensed_galaxies_z,
    Emin,
    simulate_lensed_frbs = True,
):
    """Number of FRBs per redshift step."""
    # Redshift steps
    if zlim is not None:
        zmin, zmax = zlim
    else:
        zmin = lensed_galaxies_z.min()
        zmax = lensed_galaxies_z.max()
        
    # Ensures that there are ~1000 galaxies per step on average
    z = np.linspace(zmin, zmax, lensed_galaxies_z.size//1000+1)
    # Volume steps
    V = cosmo.comoving_volume(z[1:]) - cosmo.comoving_volume(z[:-1])

    # Current FRB rate from https://ui.adsabs.harvard.edu/abs/2022arXiv220714316S/abstract
    frb_emission_rate_z0_Epivot = (7.3e4 / u.Gpc**3 / u.year).to(1 / u.Mpc**3 / u.year)
    # Rate scaled to a minimum energy Emin
    frb_emission_rate_z0 = (
        frb_emission_rate_z0_Epivot * 
        float(gammainc(gamma + 1, (Emin / Echar).value)) / 
        float(gammainc(gamma + 1, (Epivot / Echar).value))
    )
    z_mean = (z[1:] + z[:-1]) / 2
    frb_emission_rate_z = frb_emission_rate_z0 / (1 + z_mean)
    
    # Number of FRBs expected in each volume
    # FRBs rate is assumed constant with redshift
    frb_emission_rate = frb_emission_rate_z * V
    
    if simulate_lensed_frbs:
        print('Only lensed FRBs will be simulated.')
        # Fraction of lensed FRBs
        lensed_fraction = get_lensed_fraction(z)
        frb_emission_rate = frb_emission_rate * lensed_fraction
    else:
        print('Full FRB population will be simulated.')
    
    # Only keep redshift steps containing FRBs
    # This is necessary for the limit on redshift of simulated lensed galaxies
    idx_nonzero = (frb_emission_rate > 0) & ~np.isnan(frb_emission_rate)
    z_mean = z_mean[idx_nonzero]
    frb_emission_rate = frb_emission_rate[idx_nonzero]
    
    # Increased redshift resolution. Clip restricts the redshift range due to the smoothing
    z_dist = np.linspace(z.min(), z.max(), elements_in_distributions)

    # Redshift distribution
    P_z = np.interp(z_dist, z_mean, frb_emission_rate)
    frb_z = rng.choice(
        z_dist,
        size = number_of_simulated_frbs,
        p = P_z / P_z.sum()
    )
    
    # Total number of emitted FRBs per year
    total_number_of_frbs = frb_emission_rate.sum()
    return total_number_of_frbs, frb_z
    
    
def get_bandwidth_correction(
    bw_telescope,
    bw_min = 50 * u.MHz,
    loc = 400 * u.MHz,  # MHz
    scale = 400 / 2 * u.MHz,  # MHz
):
    """Assume that the bandwidth of FRBs is a normal distribution based on CHIME detections."""
    # Bandwidth distribution from a truncated normal distribution
    bw_clip = (bw_min - loc) / scale
    bw = truncnorm.rvs(
        bw_clip, 
        np.inf, 
        loc=loc, 
        scale=scale,
        size=number_of_simulated_frbs,
        random_state=rng
    ) * u.MHz
    
    # Bandwidth correction
    bw_correction = np.sqrt(bw / bw_telescope).clip(max=1)
    return bw_correction


def distribution_schechter(
    x,
    K,
    x0,
    gamma,
    normalized = False,
    xmin = None
):
    """Return the Schechter distribution."""
    y = K * (x / x0)**gamma * np.exp(-x / x0)
    if normalized:
        if xmin is None:
            scale = float(gammainc(gamma + 1))
        else:
            scale = float(gammainc(gamma + 1, xmin))
        y = y / scale
    return y


def get_energy_minimum(
    Dt,
    z,
    bw,
    sefd
):
    """Minimum FRB energy required to detect a burst at redshift z with a signal-to-noise
    ratio SNr and time resolution w, where rfi_band is lost due to interference.
    """
    Dl = cosmo.luminosity_distance(z)
    E = (
        4 * np.pi * Dl**2 / (1 + z)**(2 + alpha) * 
        np.sqrt(Dt / bw / 2) * SNr_min * sefd / np.sqrt(1 - rfi_band)
    ) * u.GHz
    return E.to(u.erg)


def get_energy_maximum(
    Emin,
    Nfrb = 0.1,
):
    """Maximum FRB energy to detect Nfrb with Emax in the survey duration above a minimum energy Emin.
    """   
    # Smaller number to avoid overflows
    x0 = Echar.to(u.erg).value / 1e41
    xmin = Emin.to(u.erg).value / 1e41
    
    # Root of the Schechter function to have Nfrb with the total FRBs emitted
    def get_schechter_root(x):
        distribution_values = distribution_schechter(x, 1/x0, x0, gamma, normalized=True, xmin=xmin/x0)
        return number_of_simulated_frbs * distribution_values - Nfrb
    sol = optimize.root(get_schechter_root, x0=x0)
    
    return sol.x[0] * 1e41 * u.erg


def get_distribution_E(
    Dt_min,
    zmin,
    bw,
    sefd
):
    # Limits of the distribution
    Emin = get_energy_minimum(
        Dt_min,
        zmin,
        bw,
        sefd
    )
    Emax = get_energy_maximum(Emin)
    # Values
    E = np.linspace(Emin, Emax, elements_in_distributions)

    # Relative probability
    distribution_E = distribution_schechter(
        E, 
        1/Echar, 
        Echar, 
        gamma, 
        xmin=Emin/Echar
    )
    # Rescaled to avoid numerical issues
    distribution_E = distribution_E / distribution_E.min()
    # Random choice accordingly to distribution
    frb_E = rng.choice(
        E,
        size = number_of_simulated_frbs,
        p = distribution_E / distribution_E.sum()
    ) * u.erg
    return frb_E, Emin


def get_beam_response(
    telescope,
    min_frb_band = 50 * u.MHz,
    efficiency = 0.7
):
    # Minimum band to detect >50 MHz of a burst
    wavelength = (telescope.v0() + min_frb_band).to(u.meter, equivalencies=u.spectral())

    try:
        # Circular aperture
        diameter = telescope.get_parameter('D')
        # Half beam width between the first nulls
        hwfn = (2.439 * u.rad * wavelength / diameter / 2).to(u.rad)
        # Random angles
        angle = rng.uniform(
            low=0,
            high=hwfn.to(u.rad).value**2,
            size=number_of_simulated_frbs
        )**0.5 * u.rad
        # Normalized power
        x = np.pi * np.sin(angle) * diameter / wavelength * np.sqrt(efficiency)
        power = (2 * j1(x) / x)**2
        # Simulated sky fraction
        sky_fraction = (hwfn**2 / 4 / u.steradian).value

    except KeyError:
        # Rectangular aperture
        size_x = telescope.get_parameter('Dx')
        size_y = telescope.get_parameter('Dy')
        # Half beam width between the first nulls
        hwfn_x = np.arcsin((wavelength / size_x).to(u.dimensionless_unscaled))
        hwfn_y = np.arcsin((wavelength / size_y).to(u.dimensionless_unscaled))
        # Random angles
        angle_x = rng.uniform(
            low=0, 
            high=hwfn_x.to(u.rad).value, 
            size=number_of_simulated_frbs
        ) * u.rad
        angle_y = rng.uniform(
            low=0, 
            high=hwfn_y.to(u.rad).value,
            size=number_of_simulated_frbs
        ) * u.rad
        # Normalized power
        power = (
            (
                np.sinc(np.sin(angle_x) * size_x / wavelength * u.rad) * 
                np.sinc(np.sin(angle_y) * size_y / wavelength * u.rad)
            )**2
        )
        # Simulated sky fraction
        sky_fraction = (hwfn_x * hwfn_y / 4 / np.pi / u.steradian).value

    return sky_fraction, power


def get_snr(
    frb_E_nu,
    frb_z,
    alpha,
    bw_telescope,
    frb_w, 
    frb_tau,
    frb_w_correction,
    frb_bw_correction,
    beam_response, 
    sefd, 
    rfi_band
):
    D_L = cosmo.luminosity_distance(frb_z)
    return (    
        (frb_E_nu * (1+frb_z)**(alpha+2)) / (4 * np.pi * D_L**2) *
        np.sqrt((2 * bw_telescope) / (frb_w + frb_tau)) * 
        frb_w_correction * frb_bw_correction *
        (beam_response / sefd) *
        np.sqrt(1 - rfi_band)
    ).to(u.dimensionless_unscaled)


# Global values

# Redshift and magnitude distributions of lensed galaxies detected by Euclid
# From https://ui.adsabs.harvard.edu/abs/2015ApJ...811...20C/abstract
lensed_galaxies_z = np.loadtxt('lenses_Euclid.txt', usecols=1)  # Redshift distribution
lensed_galaxies_Mv = np.loadtxt('lenses_Euclid.txt', usecols=17) * u.M_bol # Magnitude distribution

# FRB properties
# From https://ui.adsabs.harvard.edu/abs/2022arXiv220714316S/abstract
alpha = -1.39  # Spectral index
Echar = 2.38e41 * u.erg  # Characteristic energy cut-of
gamma = -1.3  # Differential power-law index
Epivot = 1e39 * u.erg  # Pivot energy
    
# Parameters of the simulation
number_of_simulated_frbs = int(1e5)  # Elements in the simulation
elements_in_distributions = int(1e4)  # 1 / resolution of continuous distributions
#rng = default_rng(2911167007)  # Random generator; seed fixed for reproducibility
frbs_evolve_with_luminosity = True  # FRB rate follows galaxy luminosity
zlim = None  # Limit on maximum redshift defined as [z_min, z_max]

# Assumed observational prameters
observing_time = 0.8  # Fraction of observing time
SNr_min = 8  # Minimum detectable S/N
rfi_band = 0.3  # Fraction of band affected by RFI
w_telescope = 1 * u.ms


def run_simulation(
    telescope_name = 'dummy',
    simulate_lensed_frbs = True, 
):
    """ Simulate how many strongly lensed fast radio bursts (FRBs) are detected by a facility in one year.

    Parameters
    ----------
    telescope_name : str = 'chord'
        Telescope to simulate. Currently, 'chime', 'chord', and 'dsa2000' are supported.
    simulate_lensed_frbs : bool
        Simulate only lensed FRBs or the whole population.

    Returns
    -------
    float
        Rate per year.

    """
    print(f'Starting the simulation for {telescope_name}')
    print(f'{number_of_simulated_frbs} bursts will be simulated.')
    
    # Telescope parameters
    telescope = Telescope(name=telescope_name)
    bw_telescope = 400 * u.MHz  # Subband search on constant bandwidth  #telescope.bandwidth()
    sefd = telescope.sefd()
    
    # FRB widths
    # From https://ui.adsabs.harvard.edu/abs/2021ApJS..257...59C/abstract
    sigma = 0.97
    scale = 1.0 * u.ms
    frb_w = rng.lognormal(
        sigma = sigma, 
        size = number_of_simulated_frbs
    ) * scale
    
    # FRB scattering
    # From https://ui.adsabs.harvard.edu/abs/2021ApJS..257...59C/abstract
    sigma = 1.72
    scale = 2.02 * u.ms
    frb_tau = rng.lognormal(
        sigma = sigma, 
        size = number_of_simulated_frbs
    ) * scale

    # FRB time corrections
    frb_w_correction = np.sqrt(frb_w / w_telescope).clip(max=1)
    
    # FRB bandwidth corrections
    frb_bw_correction = get_bandwidth_correction(
        bw_telescope,
        bw_min = 50 * u.MHz,
        loc = 400 * u.MHz,  # MHz
        scale = 400 / 2 * u.MHz,  # MHz
    )
    
    # FRB energies
    Dt_min = (frb_w + frb_tau).min()
    frb_E, Emin = get_distribution_E(
        Dt_min,
        lensed_galaxies_z.min(),
        bw_telescope,
        sefd
    )
    frb_E_nu = frb_E / u.GHz
    
    # Beam response
    #!!! To be corrected: now, the FoV is the same at all frequencies
    sky_fraction, beam_response = get_beam_response(telescope)
    
    # FRB redshifts
    number_of_emitted_frbs, frb_z = get_distribution_z(
        lensed_galaxies_z,
        Emin,
        simulate_lensed_frbs = simulate_lensed_frbs,
    )
    
    # FRB S/N values
    # Search single bands if present
    try:
        search_bands = telescope.get_parameter('bands')
        frb_snr = np.zeros([len(search_bands), number_of_simulated_frbs])
        for jj, (freq_low, freq_high) in enumerate(search_bands):
            frb_snr[jj] = get_snr(
                frb_E_nu,
                frb_z,
                alpha,
                bw_telescope,
                frb_w, 
                frb_tau,
                frb_w_correction,
                frb_bw_correction,
                beam_response, 
                sefd, 
                rfi_band
            )
        frb_snr = frb_snr.max(axis=0)
    except KeyError:
        frb_snr = get_snr(
            frb_E_nu,
            frb_z,
            alpha,
            bw_telescope,
            frb_w, 
            frb_tau,
            frb_w_correction,
            frb_bw_correction,
            beam_response, 
            sefd, 
            rfi_band
        )
    
    # Number of detections
    detected_frbs = frb_snr[frb_snr > SNr_min].size * observing_time / u.year
    
    # Normalize by fraction of the sky and number of FRBs simulated
    frb_detection_rate = (
        number_of_emitted_frbs * 
        detected_frbs / 
        (number_of_simulated_frbs / u.year) * 
        sky_fraction
    )
    print(f'This implies a detection rate of {frb_detection_rate.to(1/u.year).value:.1f} FRBs per year,')
    print(f'or 1 FRB detected every {1/(frb_detection_rate).to(1/u.year).value:.2f} years.')
    return frb_detection_rate

frb_detection_rate = run_simulation(telescope_name='chord', simulate_lensed_frbs = False)
