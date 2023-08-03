from scipy import constants
import numpy as np

from accphys.io import from_madx

def energy2beta0(gev):
    # Convert energy in GeV to beta0 (for electrons)
    energy_joule = gev*1e9*constants.electron_volt
    m = constants.electron_mass
    c = constants.speed_of_light
    return np.sqrt(1 - m**2*c**4/energy_joule**2)

def madx2beamline(lattice_file, gev=2.5, max_power=10, **kwargs):
    beta0 = energy2beta0(gev)
    return from_madx(lattice_file, beta0=beta0, max_power=max_power, **kwargs)

def qp2xieta(q, p):
    sqrt2 = float(np.sqrt(2))
    return (q + p*1j)/sqrt2, (q - p*1j)/sqrt2

def xieta2qp(xi, eta):
    sqrt2 = float(np.sqrt(2))
    return (xi + eta)/sqrt2, (xi - eta)/sqrt2/1j
