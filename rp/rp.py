#########################################################################
# rp.py
# Julian C. Marohnic
# Created: 5/22/23
#
# A set of functions for managing assemblies or datasets that contain
# multiple distinct rubble piles. The core capability is find_rp(), which
# returns a list of assemblies, each of which represents a single rubble
# pile. Neighbor searches are conducted using a k-D tree by default,
# although the user may substitute their own tree function. ssrp.py is
# modeled on and draws very heavily from the existing C-based rpa utility. 
# Many functions here are copied almost verbatim from rpa.c. Initially
# a standalone utility, now a secondary pkdtools module.
#
# Potential for efficiency improvements, in particular for find_rp().
#########################################################################

import numpy as np
import scipy.spatial as ss

from ..particle import Particle
from ..assembly import Assembly
from ..tools import join
from .. import aggs

# Determine whether two rubble piles meet the merger condition. From rpa.c: The following
# merger strategy is based entirely on geometry and does not take the gravitational potential
# into account. It is suitable for searching from the bottom up, that is, for starting with
# individual particles and linking them together into larger groups. The search takes the
# ellipsoidal shapes of the current groups into account. In order for rp1 to be merged with
# rp2, spheres drawn with radii equal to the major axes of the bodies (times linking-scale)
# and centered on the bodies must overlap. If the scaled minor spheres also overlap, the
# bodies are merged. Failing that, if either body has its center of mass in the other's scaled
# ellipsoid, the bodies are merged. Otherwise, no merge occurs.
#
# In contrast with rpa.c, linking-scale is passed in here as a parameter L, defaulting to 1.1.
# rp1 and rp2 must both be assemblies, containing one or more particles.
# Currently ok_to_merge() does not consider units. This shouldn't be a problem when used as a 
# utility here, but use caution...
def ok_to_merge(rp1, rp2, L):
    # Pull the necessary values from the input rubble piles.
    N1 = rp1.N()
    pos1 = rp1.com()
    semi1 = rp1.semi_axes()
    axes1 = rp1.axes()
        
    N2 = rp2.N()
    pos2 = rp2.com()
    semi2 = rp2.semi_axes()
    axes2 = rp2.axes()

    # Get major axes
    r1 = L*semi1[2]
    r2 = L*semi2[2]

    # If scaled major spheres overlap, we could have a merger.
    if overlap(pos1, pos2, r1, r2):

        # If both are singlets, we definitely have a merger.
        if N1 == 1 and N2 == 1:
            return True

        r1 = L*semi1[0]
        r2 = L*semi2[0]

        # If scaled minor spheres overlap, we definitely have a merger.
        if overlap(pos1, pos2, r1, r2):
            return True

        # Otherwise, check for one group inside the other.
        if ellipse_intersect(pos1, pos2, semi1, semi2, axes1, axes2, L):
            return True

    return False

# Indentifies closest N rubble piles to rpi.
def find_closest(rp_com, tree, N=10):
    _, nbrs = tree.query(rp_com, N)

    # First hit will be rpi itself, so discard this and take the rest.
    return nbrs[1:]

# Executes one merging pass over rubble piles.
#
# This is a little tricky. We need to loop through the list of rubble piles. In the case of a merge, we need
# to combine the two rubble piles and remove one of the merged guys so we don't duplicate. But we also don't 
# want to mess up this list structure while we're looping. The need to combine some elements and not repeat
# complicates the usual approaches. I suspect there's a better approach than what I've done here!
#
# No guarantees that this will work nicely for anything other than a "good faith" rubble pile!
# This is currently very slow, the loop/list structure here is the bottleneck.
def merge_rp(rp_list, locs, tree, L, units=None):
    # We need to specify units because join defaults to 'pkd'.
    if units != None and units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")
    if units == None:
        units = 'pkd'

    Nrp = len(rp_list)
    nMerge = 0
    new_list = []

    # Check whether the main rp has been merged already. If it has, move to the next one. If not, check for a
    # possible merge with nearest neighbor. Ensure nearest neighbor has not been merged yet.
    for i, rp in enumerate(rp_list):
        if rp == None:
            continue

        rp_com = locs[i]
        nbr_indices = find_closest(rp_com, tree, min(4, Nrp))

        for nbr_index in nbr_indices:
            nbr = rp_list[nbr_index] 

            if nbr != None and ok_to_merge(rp, nbr, L):
                rp_list[i] = join(rp, nbr, units=units)
                rp_list[nbr_index] = None
                nMerge += 1
                break

    rp_list = [rp for rp in rp_list if rp != None]
    return nMerge, rp_list

# Currently expects a list of assemblies representing the current set of rubble piles.
def build_tree(rp_list, tree_func):
    # Build list of all rp centers
    coms = [assembly.com() for assembly in rp_list]

    return coms, tree_func(coms)


# Returns True if a ball with radius R at r0 lies entirely within the ellipsoid defined by
# semi-axes 'a' (measured along the Cartesian axes and centered at the origin). To get this
# right, need to compute direction from r0 to nearest point on ellipsoid surface. This is too
# hard, so settle for more conservative boundary. Copied from rpu.c more or less verbatim.
def in_ellipsoid(r0, R, a):
    if check_vector(a) == 0:
        raise TypeError("a must be a real-valued 3-vector.")
    if a[0] <= 0.0 or a[1] <= 0.0 or a[2] <= 0.0:
        raise ValueError("All semi-axes must be positive.")

    # Convert to numpy arrays for easier manipulation.
    r0 = np.array(r0)
    a = np.array(a)
    
    # Initialize temp variable and check easy cases.

    d = 0

    for i in range(3):
        if r0[i] == 0 and R > a[i]:
            return False
        d += (r0[i]/a[i])**2

    # Centered at the origin and we know from the previous check in the for
    # loop above that if it is centered at the origin, it fits.
    if d == 0:
        return True
    # d > 1 guarantees at least 1 coordinate of the center is at or beyond
    # the edge of the ellipsoid in the correseponding axis.
    if d > 1:
        return False

    # Make a conservative guess for the general case.

    d = np.linalg.norm(r0)
    r = r0.copy()
    r = r*(1 + R/d)
    d = (r[0]/a[0])**2 + (r[1]/a[1])**2 + (r[2]/a[2])**2

    if d <= 1:
        return True
    else:
        return False

# This is really rough, should clean this up when there's time.
def ellipse_intersect(pos1, pos2, semi1, semi2, axes1, axes2, L):
    if check_vector(semi1) == 0:
        raise TypeError("semi1 must be a real-valued 3-vector.")
    if check_vector(semi2) == 0:
        raise TypeError("semi2 must be a real-valued 3-vector.")
    if semi1[0] <= 0.0 or semi1[1] <= 0.0 or semi1[2] <= 0.0:
        raise ValueError("All semi-axes must be positive.")
    if semi2[0] <= 0.0 or semi2[1] <= 0.0 or semi2[2] <= 0.0:
        raise ValueError("All semi-axes must be positive.")

    # Convert to numpy arrays for easier manipulation.
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    semi1 = np.array(semi1)
    semi2 = np.array(semi2)
    R1 = np.mean(semi1)
    R2 = np.mean(semi2)

    # Check for ellipse 2 in ellipse 1 
    r0 = pos2 - pos1
    r0 = np.dot(axes1, r0)

    d = np.linalg.norm(r0)
    r = r0.copy()
    # Could be that ellipse 2 is way bigger, in which case we could get a negative value here. If that's the case
    # just say 0.
    r = max(1 - L*R2/d, 0.0)*r
    d = (r[0]/semi1[0])**2 + (r[1]/semi1[1])**2 + (r[2]/semi1[2])**2

    if d <= 1:
        return True
    
    # Check for ellipse 1 in ellipse 2
    r0 = pos1 - pos2
    r0 = np.dot(axes2, r0)

    d = np.linalg.norm(r0)
    r = r0.copy()
    r = max(1 - L*R1/d, 0.0)*r 
    d = (r[0]/semi2[0])**2 + (r[1]/semi2[1])**2 + (r[2]/semi2[2])**2

    if d <= 1:
        return True
    else:
        return False

# Determine whether two spheres are overlapping, given positions and radii.
def overlap(pos1, pos2, R1, R2):
    if check_vector(pos1) == 0 or check_vector(pos2) == 0:
        raise TypeError("pos1 and pos2 must be real-valued 3-vectors.")

    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    disp = np.linalg.norm(pos1 - pos2)

    if disp <= (R1 + R2):
        return True
    else:

        return False

# Determine if element is a real-valued 3-vector. Return 1 if it is, 0 if not.
def check_vector(element):
    if not isinstance(element, np.ndarray) and not isinstance(element, tuple) and not isinstance(element, list):
        return 0
    if len(element) != 3:
        return 0
    for component in element:
        if type(component) not in [int, float, np.int64, np.float64]:
            return 0

    return 1

# Determine if element is an assembly.
def check_assembly(element):
    if isinstance(element, Assembly):
        return 1
    else:
        return 0

# Take a standard assembly and return a list of assemblies, each a singlet containing one of the original particles.
def split_particles(self, units=None):
    if units == None:
        units = self.units
    if units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")

    split = []

    for particle in self:
        split.append(Assembly(particle.copy(), units=units))

    return split
Assembly.split_particles = split_particles

# Assembly method for pulling out distinct rubble piles.
def find_rp(self, L=1.1, tree_func=ss.KDTree, units=None):
    # Ensure consistent units. Keep everything in terms of input assembly's units. Most function called will do this
    # automatically, but I prefer to be explicit.
    if units != None and units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")
    if units == None:
        units = self.units

    # Initialize rubble pile list in the case of no agg merge. This places each particle in the assembly into its
    # own individual assembly in preparation for merges.
    rp_list = self.split_particles(units=units)

    nMerge = 1

    while nMerge > 0:
        # Exit loop if all particles end up in a single assembly.
        if len(rp_list) == 1:
            break

        # Build tree using rp CoMs.
        coms, tree = build_tree(rp_list, tree_func)
        nMerge, rp_list = merge_rp(rp_list, coms, tree, L, units)

    # Sort rp's in descending order by mass before returning.
    return sorted(rp_list, key=lambda rp: rp.M(), reverse=True)
Assembly.find_rp = find_rp

# Calculate the elongation of a rubble pile assembly.
def elong(self):
    semi = self.semi_axes()
    alpha1 = semi[0]/semi[2]
    alpha2 = semi[1]/semi[2]

    # Calculating elongation as Yun does in her 2020 paper.
    elong = 1 - (alpha1 + alpha2)/2
    return elong
Assembly.elong = elong

# Add helper functions to get statistics
    # spins?
    # shape metric (finally...)

# Functions intended to be applied to a list of assemblies, each of which is a distinct
# rubble pile (like that returned by find_rp()).

# Return total mass of the collection of rp's.
def tot_M(rp_list):
    return sum([rp.M() for rp in rp_list])

# Returns mass and mass fraction of most massive fragment/rubble pile.
def max_M(rp_list):
    masses = [rp.M() for rp in rp_list]
    total = sum(masses)
    max_M = max(masses)
    frac = max_M/total

    return max_M, frac

# Calculates total number of fragments in the list of rubble piles. This probably doesn't warrant its own function,
# but I'm including it for consistency since this is a metric we might care about.
def N_frags(rp_list):
    return len(rp_list)

# Return a new list of rp's with any "singlets" removed. I.e., lone particles or lone aggs. Changing the 'particles' or 'aggs' 
# inputs to False will toggle OFF this behavior. Can also remove all fragments/rp's below a given mass fraction (off by default).
def rm_single(rp_list, particles=True, aggs=True, mass_frac=False, min_num=False):
    if not isinstance(particles, bool) or not isinstance(aggs, bool):
        raise TypeError("'particles' and 'aggs' arguments must be Boolean values.")
    if mass_frac != False:
        if not isinstance(mass_frac, float) or not (0 < mass_frac < 1):
            raise ValueError("'mass_frac' argument must be either False or a number between 0 and 1.")
    if min_num != False:
        if not isinstance(min_num, int) or not (0 < min_num):
            raise ValueError("'min_num' argument must be either False or a number greater than 0.")

    rp_list = rp_list.copy()

    M = tot_M(rp_list)
    new_rps = []

    for rp in rp_list:
        fail = 0

        if particles == True and rp.N() == 1:
            fail = 1
        elif aggs == True and is_agg(rp) == True:
            fail = 1
        elif mass_frac != False and rp.M()/M < mass_frac:
            fail = 1
        elif min_num != False and rp.N() < min_num:
            fail = 1
            
        if fail == 1:
            continue
        else:
            new_rps.append(rp)

    return new_rps

# Return True if the assembly passed is a single aggregate (i.e., exclusively contains particles with the same, negative
# iOrgIdx value). Used in rm_single().
def is_agg(assembly):
    if not isinstance(assembly, Assembly):
        raise TypeError("is_agg may only be called on an Assembly.")

    # Get the iOrgIdx of the first particle in the assembly for comparison. If it's nonnegative, we fail immediately.
    base_tag = assembly[0].iOrgIdx
    if base_tag >= 0:
        return False

    for particle in assembly:
        if particle.iOrgIdx != base_tag:
            return False

    return True
