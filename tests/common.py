from scipy import constants
import numpy as np

from accphys import file2dataframe, to_beamline

def energy2beta0(gev):
    # Convert energy in GeV to beta0 (for electrons)
    energy_joule = gev*1e9*constants.electron_volt
    m = constants.electron_mass
    c = constants.speed_of_light
    return np.sqrt(1 - m**2*c**4/energy_joule**2)

def madx2beamline(lattice_file, gev=2.5, max_power=10):
    beta0 = energy2beta0(gev)
    raw_df, inp = file2dataframe(lattice_file)
    seq = to_beamline(raw_df, beta0=beta0, max_power=max_power, **inp)
    return seq

def qp2xieta(q, p):
    sqrt2 = float(np.sqrt(2))
    return (q + p*1j)/sqrt2, (q - p*1j)/sqrt2

def xieta2qp(xi, eta):
    sqrt2 = float(np.sqrt(2))
    return (xi + eta)/sqrt2, (xi - eta)/sqrt2/1j
