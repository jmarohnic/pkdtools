##################################################################
# dgrid.py 
# Julian C. Marohnic
# Created: 6/5/23
#
# A set of functions to support analysis of large suites of tidal
# disruption or spinup runs (or anything other process producing
# lots of rubble pile fragments). Uses a 'DataGrid' class
# that makes uses of the pandas DataFrame to hold information about
# a given suite of runs.
##################################################################

import numpy as np
import os

import pandas as pd
import subprocess as sp

from ..assembly import Assembly
from ..pkdio import ss_in
from .. import rp

# A new class intended to accumulate data as walk_grid() progresses through a given tidal suite of runs.
# DataGrid should be provided with a list of vinf values, a list of q values, a filename, and an analysis function.
# This function should ingest a list of rubble piles (i.e., output from find_rp()) and return a single
# value for a given run.
# NEW IDEA: Allow function to operate on a directory rather than just fragments. We may care about entities other 
# than fragments.
class DataGrid:
    def __init__(self, q_list, vinf_list, filename, func):
        self.q_list = q_list
        self.vinf_list = vinf_list
        self.filename = filename
        self.func = func
        self.grid = pd.DataFrame(columns=vinf_list, index=q_list)

    def read(self, q, vinf):
        return self.grid.at[q, vinf]

    def write(self, q, vinf, val):
        self.grid.at[q, vinf] = val

    def save_csv(self):
        self.grid.to_csv(self.filename)

    def save_pickle(self):
        self.grid.to_pickle(self.filename)

# Step through each run and call the desired analysis functions on the last file.
# 'fragrids' should be a list of DataGrids for analyses that take a list of all fragments as input.
# 'othergrids' should be a list of DataGrids for all analyses that do not meet this format, and
# require a more general initialization.
def walk_grid(q_list, vinf_list, filename=None, fraggrids=[], stepgrids=[], othergrids=[], L=1.2, fragunits='pkd', stepunits='pkd', otherunits='pkd'):
    for vinf in vinf_list:
        for q in q_list:
            dir = f"q{q:.1f}_vinf{vinf}"
            if os.path.exists(dir):
                print(f"Entering {dir}")
                os.chdir(dir)
                if fraggrids != []:
                    analyze_frags(q, vinf, *fraggrids, filename=filename, L=L, units=fragunits)
                if stepgrids != []:
                    analyze_steps(q, vinf, *stepgrids, units=stepunits)
                if othergrids != []:
                    analyze_other(q, vinf, *othergrids, units=otherunits)
                os.chdir("..")
            else:
                print(f"{dir} not found, skipping.")

    # Write each grid to a file.
    if fraggrids != []:
        for grid in fraggrids:
            grid.save_csv()
    if stepgrids != []:
        for grid in stepgrids:
            grid.save_csv()
    # "Other" grids are saved in Python pickle format rather than csv to accomodate non-numeric data like histograms.
    if othergrids != []:
        for grid in othergrids:
            grid.save_pickle()

# Load data, find rp's and clean up Earth particle. Any further fragment cleaning should happen in the
# analysis function, since this may be treated differently for different metrics. Grouping this way avoids 
# calling find_rp() for each individual analysis. Any datagrids passed in will be updated. Nothing should be returned.
def analyze_frags(q, vinf, *fraggrids, filename=None, L=1.2, del_earth=True, units='pkd'):
    # Intialize all grids with NaN value.
    for grid in fraggrids:
        grid.write(q, vinf, float("nan"))

    # Check filename input. If this argument is not a string or is left empty, analyze_frags() will automatically find
    # the last output file in the run directory.
    if isinstance(filename, str):
        pass
    else:
        filename = final_step(get_stepsize(), get_zeropad())

    # If all is well, load desired output file.
    if os.path.exists(filename):
        ssdata = ss_in(filename, units=units)
    else:
        return

    if del_earth == True:
        ssdata = rm_earth(ssdata)

    frags = ssdata.find_rp(L=L)

    for grid in fraggrids:
        value = grid.func(frags)
        grid.write(q, vinf, value)

# Use for analyses that require stepping through all (or a subset of) output files, rather than considering only
# the final state. E.g., tidal_threshold().
def analyze_steps(q, vinf, *stepgrids, del_earth=True, units='pkd'):
    # Intialize all grids with NaN value.
    for grid in stepgrids:
        grid.write(q, vinf, float("nan"))

    stepsize = get_stepsize()
    zeropad = get_zeropad()
    steplist = get_steps(stepsize, zeropad)

    for grid in stepgrids:
        value = grid.func(q, vinf, steplist, units=units)
        grid.write(q, vinf, value)

# We set aside calls to any functions that *do not* require fragment analysis, since
# find_rp() is expensive and we benefit substantially from only performing this calculation once per run.
# Any analysis functions that don't fit the final state fragment analysis model well can be called here. 
# No assumptions about inputs are made beyond q and vinf.
def analyze_other(q, vinf, *othergrids, units='pkd'):
    # Intialize all grids with NaN value.
    for grid in othergrids:
        grid.write(q, vinf, float("nan"))

    for grid in othergrids:
        value = grid.func(q, vinf, units=units)
        grid.write(q, vinf, value)

# A much more general approach to traversing a suite of tidal encounter-style runs, in contrast to walk_grid().
# Accepts q and vinf lists to be treated as a grid a function to be called with no arguments. Anything can
# go in this function.
def gen_walk(q_list, vinf_list, func):
    for vinf in vinf_list:
        for q in q_list:
            dir = f"q{q:.1f}_vinf{vinf}"
            if os.path.exists(dir):
                print(f"Entering {dir}")
                os.chdir(dir)
                func()
                os.chdir("..")
            else:
                print(f"{dir} not found, skipping.")




#### FRAG GRIDS ####

# Returns the mass fraction of the most massive fragment in the list of rubble piles.
def largest_fragment_fraction(frags):
    M = rp.tot_M(frags)
    clean = rp.rm_single(frags, min_num=10)

    frac_masses = [rp.M()/M for rp in clean]
    return max(frac_masses)

# Returns the elongation of the most massive fragment.
def largest_fragment_elongation(frags):
    clean = rp.rm_single(frags, min_num=10)

    biggest = clean[0]
    return biggest.elong()

# Returns the total number of fragments, after removing anything under 10 particles.
def N_fragments(frags):
    clean = rp.rm_single(frags, min_num=10)
    return len(clean)

# Walks through each fragment (above some threshold size) and determines whether the
# fragment is bound to the largest fragment via simple two-body calculation. Returns
# total mass *bound* to core fragment.
def largest_fragment_bound(frags):
    # Important for calculation of U below, which uses a fixed value for G. This should be improved in
    # the future when unit-ful fundamental constants are available from ssedit/pkdtools. Only consistency
    # matter here since final return value is unitless.
    for frag in frags:
        frag.units = 'mks'

    M = rp.tot_M(frags)

    # No cleaning for now: consider all mass.
    #clean = rp.rm_single(frags, min_num=10)
    clean = frags
    if len(clean) == 0:
        return float("nan")
    if len(clean) == 1:
        return 1.0

    # Should be ordered in descending mass order at this point.
    large = clean[0]
    largeM = large.M()
    largecom = large.com()
    largev = large.comv()

    # Mass of largest fragment is already known to be "bound" here.
    bound_mass = largeM

    for frag in clean[1:]:
        # Calculate kinetic energy of fragment relative to core piece.
        fragM = frag.M()
        fragcom = frag.com()
        fragv = frag.comv()
        fragK = 0.5*fragM*np.linalg.norm(fragv - largev)**2

        # Calculate relative gravitational potential.
        U = -6.6743e-11*largeM*fragM/np.linalg.norm(fragcom - largecom)

        # If relative total energy is negative, add fragment's mass to total of bound mass.
        if fragK + U < 0:
            bound_mass += fragM

    return bound_mass/M

# Return the fraction of total progenitor mass that ends up in orbit about the largest fragment.
# Contrast with largest_fragment_bound(), which includes *all* mass gravitationally bound with
# that largest fragment, including the fragment mass itself. Here, we only consider mass that is
# in orbit about the large fragment.
def largest_fragment_orbiting(frags):
    # Important for calculation of U below, which uses a fixed value for G. This should be improved in
    # the future when unit-ful fundamental constants are available from ssedit/pkdtools. Only consistency
    # matter here since final return value is unitless.
    for frag in frags:
        frag.units = 'mks'

    M = rp.tot_M(frags)

    # No cleaning for now: consider all mass.
    #clean = rp.rm_single(frags, min_num=10)
    clean = frags
    if len(clean) == 0:
        return float("nan")

    # Should be ordered in descending mass order at this point.
    large = clean[0]
    largeM = large.M()
    largecom = large.com()
    largev = large.comv()

    # We ignore the mass of the largest fragment here, otherwise very similar to largest_fragment_bound().
    orbit_mass = 0

    for frag in clean[1:]:
        # Calculate kinetic energy of fragment relative to core piece.
        fragM = frag.M()
        fragcom = frag.com()
        fragv = frag.comv()
        fragK = 0.5*fragM*np.linalg.norm(fragv - largev)**2

        # Calculate relative gravitational potential.
        U = -6.6743e-11*largeM*fragM/np.linalg.norm(fragcom - largecom)

        # If relative total energy is negative, add fragment's mass to total of bound mass.
        if fragK + U < 0:
            orbit_mass += fragM

    return orbit_mass/M

# Return the spin period of the largest post-encounter fragment in units of hours.
def largest_fragment_period(frags):
    # Set units to mks to get a period in seconds. Convert to hours at the end.
    for frag in frags:
        frag.units = 'mks'
    
    if len(frags) == 0:
        return float("nan")

    large = frags[0]
    period = large.period()

    # Factor of 3600 accounts for seconds --> hours unit conversion.
    return period/3600.




#### STEP GRIDS ####

# Return the fraction of material in the largest fragment that came from the inner 50% of the initial body by *volume*.
def largest_fragment_volcore(q, vinf, steplist, units='pkd'):
    # Load initial step data and and calculate CoM and radius.
    init = ss_in(steplist[0], units=units)
    init = rm_earth(init)
    initR = init.R()
    initcom = init.com()
    depth_dict = {}

    # Assign depth values of 0 for "surface" material and 1 for "core" material to build a "depth dictionary" for particles.
    for particle in init:
        # The coefficient on initR is the fraction of the radius you need to split the volume of a sphere in two, assuming a uniform
        # density distribution.
        if np.linalg.norm(particle.pos() - initcom) < (1/(2**(1/3)))*initR:
            depth_dict[particle.iOrder] = 1
        else:
            depth_dict[particle.iOrder] = 0

    # Split final output into fragments. Calculate fraction of largest fragment that has a "core" tag, using core_total as a counter.
    final = ss_in(steplist[-1], units=units)
    final = rm_earth(final)
    final_frags = final.find_rp(L=1.2)
    large = final_frags[0]
    core_total = 0

    for particle in large:
        if depth_dict[particle.iOrder] == 1:
            core_total += 1

    return core_total/large.N()

# Return the fraction of material in the largest fragment that came from the inner 50% of the initial body by *radius*. Identical to
# largest fragment_volcore, but with a different criterion for "core". Here, any particle closer to the center than half of the radius
# is considered core. This is a more restrictive condition, so we should expect lower values.
def largest_fragment_radcore(q, vinf, steplist, units='pkd'):
    # Load initial step data and and calculate CoM and radius.
    init = ss_in(steplist[0], units=units)
    init = rm_earth(init)
    initR = init.R()
    initcom = init.com()
    depth_dict = {}

    # Assign depth values of 0 for "surface" material and 1 for "core" material to build a "depth dictionary" for particles.
    for particle in init:
        if np.linalg.norm(particle.pos() - initcom) < 0.5*initR:
            depth_dict[particle.iOrder] = 1
        else:
            depth_dict[particle.iOrder] = 0

    # Split final output into fragments. Calculate fraction of largest fragment that has a "core" tag, using core_total as a counter.
    final = ss_in(steplist[-1], units=units)
    final = rm_earth(final)
    final_frags = final.find_rp(L=1.2)
    large = final_frags[0]
    core_total = 0

    for particle in large:
        if depth_dict[particle.iOrder] == 1:
            core_total += 1

    return core_total/large.N()

# Need to write up a better version of this---get a better handle on units first. Currently returns a value in km regardless of units input.
def tidal_threshold(q, vinf, steplist, units='pkd'):
    # Set "disruption threshold" of 0.5% of radius.
    threshold = 1.005

    # Determine radius of initial progenitor body.
    init = ss_in(steplist[0], units=units)
    init = rm_earth(init)
    init_R = init.R()

    for filename in steplist:
        current = ss_in(filename, units=units)
        earth = pop_earth(current)
        R = current.R()
        
        if R > threshold*init_R:
            distance = np.linalg.norm(current.com() - earth.pos())
            return distance

    return float("nan")

# Determine if a run is done evolving.
def tidal_run_complete(q, vinf, steplist, units='pkd'):
    L = 3.5
    # Load initial and final step data, divide into fragments and calculate the mass fraction bound to the largest piece.
    init = ss_in(steplist[0], units=units)
    init = init.rm_earth()
    init_frags = init.find_rp(L=L)
    init_bound_frac = largest_fragment_bound(init_frags)

    final = ss_in(steplist[-1], units=units)
    final = final.rm_earth()
    final_frags = final.find_rp(L=L)
    final_bound_frac = largest_fragment_bound(final_frags)

    # Every tidal run should have an associated sstidal.log file, which includes a calculated periapse time.
    periapse_time = float(sp.check_output("grep 'Time to periapse' sstidal.log | awk '{print $5}'", shell=True))

    # Quick first check. If run has not proceeded past 3x the periapse time, instant fail.
    if final.time < 3*abs(periapse_time):
        return False

    # Not a great approach, may need to iterate on this.
    # IDEA: We've ensured we're well past periapse. At this point, either bound mass ratio should be unchanged, or
    # it should have *stopped* changing.

    if final_bound_frac > 0.95:
        return True

    # Final bound mass fraction is significantly different from initial. Check for stability, comparing to bound mass fraction from 5
    # output prior to final output. This will depend on iOutInterval setting, but should be good enough for now.
    prev = ss_in(steplist[-10], units=units)
    prev = prev.rm_earth()
    prev_frags = prev.find_rp(L=L)
    prev_bound_frac = largest_fragment_bound(prev_frags)

    if (final_bound_frac < 0.9*prev_bound_frac) or (final_bound_frac > 1.1*prev_bound_frac):
        return False
    else:
        return True




#### UTILITY ####

# Extract the iOutInterval value from ss.par
def get_stepsize():
    if os.path.exists("ss.par"):
        stepsize = sp.check_output("grep iOutInterval ss.par | awk '{print $3}'", shell=True)
        return int(stepsize)
    else:
        raise ValueError("No ss.par file found, cannot determine stepsize.")

# Extract the nDigits value from ss.par
def get_zeropad():
    if os.path.exists("ss.par"):
        zeropad = sp.check_output("grep nDigits ss.par | awk '{print $3}'", shell=True)
        return int(zeropad)
    else:
        raise ValueError("No ss.par file found, cannot determine ss file digit mask.")

# Find the final sequential ss output file in the current directory.
def final_step(stepsize, zeropad):
    steps = get_steps(stepsize, zeropad)
    return steps[-1]

# Return a list of all sequential ss output files in the current directory, including init.
def get_steps(stepsize, zeropad, init="initcond.ss"):
    step_num = stepsize
    cur_step = f"ss.{0:0{zeropad}}"
    next_step = f"ss.{step_num:0{zeropad}}"

    all_steps = [init]

    while os.path.exists(next_step):
        step_num += stepsize
        cur_step = next_step
        next_step = f"ss.{step_num:0{zeropad}}"
        all_steps.append(cur_step)        

    return all_steps

# Get rid of the earth particle, if present.
def rm_earth(assembly):
    copy = assembly.copy()
    copy.units = 'pkd'

    earth_id = [particle.iOrder for particle in copy if particle.m > 3e-6]

    if len(earth_id) > 1:
        raise ValueError("Assembly has more than one Earth candidate particle.")
    elif len(earth_id) == 0:
        print("No Earth particle found. No changes made.")
        return assembly
    else:
        earth_id = earth_id[0]

    # Remove earth particle and restore input assembly units.
    copy.del_particles(earth_id)
    copy.units = assembly.units
    return copy
Assembly.rm_earth = rm_earth

# Pop Earth particle from assembly, if present.
def pop_earth(assembly):
    copy = assembly.copy()
    copy.units = 'pkd'

    earth_id = [particle.iOrder for particle in copy if particle.m > 3e-6]

    if len(earth_id) > 1:
        raise ValueError("Assembly has more than one Earth candidate particle.")
    elif len(earth_id) == 0:
        raise ValueError("Assembly has no Earth particle.")
    else:
        earth_id = earth_id[0]

    earth = copy.get_particle(earth_id)
    earth.units = assembly.units
    assembly.del_particles(earth_id)

    return earth
Assembly.pop_earth = pop_earth
