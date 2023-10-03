##################################################################
# hist.py 
# Julian C. Marohnic
# Created: 9/19/23
#
# Tools and a custom class for calculating and collecting histograms
# across tidal encounter-style sets of simulations.
##################################################################

import numpy as np

from ..pkdio import ss_in

from .dgrid import rm_earth
from .dgrid import get_stepsize
from .dgrid import get_zeropad
from .dgrid import final_step

class Histogram:
    def __init__(self, counts, bin_edges):
        self.counts = counts
        self.bin_edges = bin_edges

# Find the final timestep and calculate histogram data of fragment mass
# as a function of total system mass.
def get_mass_hist(q, vinf, units='pkd'):
    # Pick out final output file.
    filename = final_step(get_stepsize(), get_zeropad())
    ssdata = ss_in(filename, units)
    ssdata = rm_earth(ssdata)

    M = ssdata.M()
    frags = ssdata.find_rp(L=1.2)
    # Calculate mass fraction for all fragments.
    frag_mass = [frag.M()/M for frag in frags] 

    counts, bin_edges = np.histogram(frag_mass, bins=15, range=(0,1))

    return Histogram(counts, bin_edges)

# Find the final timestep and return a sorted list of all fragment
# masses as a function of total system mass.
def get_frag_masses(q, vinf, units='pkd'):
    # Pick out final output file.
    filename = final_step(get_stepsize(), get_zeropad())
    ssdata = ss_in(filename, units)
    ssdata = rm_earth(ssdata)

    M = ssdata.M()
    frags = ssdata.find_rp(L=1.2)
    # Calculate mass fraction for all fragments.
    frag_masses = [frag.M()/M for frag in frags] 

    return frag_masses
