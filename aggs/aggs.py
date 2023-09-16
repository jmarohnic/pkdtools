##################################################################
# aggs.py
# Julian C. Marohnic
# Created: 8/16/23
# Adapted from earlier work (c. 2022-23) by JCM and JVD.
#
# A set of useful methods for aggregates and functions for generating
# the regular aggregate shapes: dumbbell, diamond, rod, tetrahedron,
# and cube.
##################################################################

import numpy as np

from ..particle import Particle
from ..assembly import Assembly

# Find agg with largest (negative) index.
def agg_max(self):
    return min([particle.iOrgIdx for particle in self])
Assembly.agg_max = agg_max

def agg_min(self):
    return max([particle.iOrgIdx for particle in self if particle.iOrgIdx < 0])
Assembly.agg_min = agg_min

def agg_range(self):
    agg_tags = [particle.iOrgIdx for particle in self if particle.iOrgIdx < 0]
    return (max(agg_tags), min(agg_tags))
Assembly.agg_range = agg_range

def agg_list(self):
    agg_list = np.unique([particle.iOrgIdx for particle in self if particle.iOrgIdx < 0])

    # Determine whether the assembly contains any aggregates.
    if len(agg_list) == 0:
        print("No aggs in this assembly.")
        return None
    else:
        # Return reversed agg_list, starting with -1.
        return agg_list[::-1]
Assembly.agg_list = agg_list

# Return number of aggs in the assembly.
def N_aggs(self):
    return len(self.agg_list())
Assembly.N_aggs = N_aggs

# Returns a new assembly consisting only of particles in the desired aggregate.
def get_agg(self, iOrgIdx):
    if not isinstance(iOrgIdx, int):
        raise TypeError("Warning: get_agg() takes a single negative integer as its argument.")
    if iOrgIdx >= 0:
        raise ValueError("Warning: get_agg() takes a single negative integer as its argument.")

    matches = [particle for particle in self if particle.iOrgIdx == iOrgIdx]
    return Assembly(*matches, units=self.units, time=self.time)
Assembly.get_agg = get_agg

# Delete specified aggs from the assembly.
def del_aggs(self, *iOrgIdxs):
    for element in iOrgIdxs:
        if not isinstance(element, int):
            raise TypeError("Warning: del_agg() takes negative integers as arguments.\n"
                            "If you would like to use a list to specify the aggs to be deleted, use the '*' operator.\n")
        if element >= 0:
            raise ValueError("Warning: del_agg() takes negative integers as arguments.\n"
                            "If you would like to use a list to specify the aggs to be deleted, use the '*' operator.\n")
    del_list = [particle.iOrder for particle in self if particle.iOrgIdx in iOrgIdxs]
    self.del_particles(*del_list)
Assembly.del_aggs = del_aggs

# "Pop" the desired agg from the assembly, deleting it from the assembly and returning a new copy.
def pop_agg(self, iOrgIdx):
    if not isinstance(iOrgIdx, int):
        raise TypeError("Warning: pop_agg() takes a single negative integer as its argument.")
    if iOrgIdx >= 0:
        raise ValueError("Warning: pop_agg() takes a single negative integer as its argument.")

    del_list = [particle.iOrder for particle in self if particle.iOrgIdx == iOrgIdx]
    matches = [particle for particle in self if particle.iOrgIdx == iOrgIdx]

    new = Assembly(*matches, units=self.units, time=self.time)
    self.del_particles(*del_list)

    return new
Assembly.pop_agg = pop_agg

# Find any single particles with iOrgIdx < 0 ("orphans") and set iOrgIdx = iOrder.
# Currently very slow. Consider ways to make this operation more efficient.
def fix_orphans(self):
    agg_tags = [particle.iOrgIdx for particle in self if particle.iOrgIdx < 0]
    orphans = []
    for index in agg_tags:
        if agg_tags.count(index) == 1:
            orphans.append(index)

    if len(orphans) == 0:
        print("No orphan particles found in this assembly.")
        return None

    for particle in self:
        if particle.iOrgIdx in orphans:
            particle.iOrgIdx = particle.iOrder

    print(len(orphans), "orphan(s) corrected.")
Assembly.fix_orphans = fix_orphans

# Renumbers negative iOrgIdxs consecutively, keeping aggregates together
def condense_aggs(self, direction='d'):
    self.fix_orphans()

    iAggIdxCounter = decrement = 0
    self.sort_iOrgIdx()
    for particle in self:
        particle.iOrgIdx -= decrement
        if particle.iOrgIdx < 0 and particle.iOrgIdx != iAggIdxCounter:
            iAggIdxCounter -= 1
            if particle.iOrgIdx != iAggIdxCounter:
                change = particle.iOrgIdx - iAggIdxCounter
                decrement += change
                particle.iOrgIdx -= change

    if direction == 'a':
        self.sort_iOrgIdx('a')
Assembly.condense_aggs = condense_aggs

# Return a list of assemblies, each containing one agg from the assembly passed in. Units match those of
# the assembly argument unless an alternative is specified.
def all_aggs(self, units=None):
    if units == None:
        units = self.units
    # Check for valid specification of units.
    if units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")

    # Get all negative iOrgIdx values.
    agg_list = self.agg_list()

    # Create a new assembly for each agg and add it to the list.
    aggs = []
    for i in agg_list:
        aggs.append(Assembly(*[particle for particle in self if particle.iOrgIdx == i], units=units))

    return aggs
Assembly.all_aggs = all_aggs

##### FUNCTIONS FOR GENERATING REGULAR AGGREGATE SHAPES #####

# Generate an assembly consisting of a single dumbbell-shaped aggregate. User may specify a mass and "radius" for the whole agg, 
# or for each particle (using the pmass and pradius arguments in lieu of mass and radius). Suggest passing keyword arguments only to avoid confusion. 
# Specifying orientation in this way obviously leaves some degeneracy in the final attitude of the agg but gives the user some degree of
# control. This may be improved in the future.
def make_db(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=2, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    # Ensure that user supplies at least one set of mass/radius arguments.
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    # Make sure user doesn't overconstrain the agg.
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    # Avoiding overlap between 'units' as an argument of this function and 'units' as an Assembly attribute.
    temp_units = units

    # If user has specified agg mass or radius, set *particle* mass and radius (pmass and pradius) accordingly and continue.
    # Aggregate "radius" is taken to be the maximum distance between agg center and a particle edge. The "separation" s (or "sep")
    # is given by sep_coeff*pradius and is taken to be the distance between particle centers in the dumbbell case.
    # Need to account for this when converting between agg radius and particle radius.
    if mass > 0 or radius > 0:
        pmass = mass/2
        pradius = radius*(1 + sep_coeff/2)**(-1)

    # Set particle separation for placement within the agg and set relative center positions. These will be translated
    # by the specified agg center location after the desired orientation is applied.
    sep = sep_coeff*pradius
    p0_center = np.array([0,0,-sep/2])
    p1_center = np.array([0,0,+sep/2])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)

    agg = Assembly(p0, p1, units=temp_units)

    # Set orientation. (0,0,1) is the default. Again, note that particle centers are translated *after* agg is rotated.
    # Order of operations matters here! Check first if rotation is necessary to avoid annoying warnings from rotate() method.
    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        # Normalize axis.
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg

# Generate a planar diamond-shaped aggregate.
def make_diamond(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=12, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    # In the case of planar diamonds, the relation between particle radius and agg radius is different. Agg radius is s + pradius.
    if mass > 0 or radius > 0: 
        pmass = mass/4
        pradius = radius*(1 + sep_coeff)**(-1)

    sep = sep_coeff*pradius
    p0_center = np.array([0,0,-sep])
    p1_center = np.array([0,0,+sep])
    p2_center = np.array([-sep/2,0,0])
    p3_center = np.array([+sep/2,0,0])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg

# Generate a tetrahedron-shaped aggregate. This function generates a tetrahedron with one flat face on the bottom and an upright
# pyramid-like orientation. This is in contrast with genTetrahedron() in ssgen2Agg.py, which uses a much more elegant formulation
# for a tetrahedron with 2 level edges, but lacking an upright orientation. Since we care about orientation here, we use the former
# approach. Default position should have centroid at the origin.
def make_tetra(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=3, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    if mass > 0 or radius > 0: 
        pmass = mass/4
        pradius = radius*(1 + sep_coeff/2)**(-1)

    # Set particle separation for placement within the agg and set relative center positions. These will be translated
    # by the specified agg center location after the desired orientation is applied. Given the weird coordinates needed for
    # laying out the tetrahedron, we specify the locations of vertices on the unit circle as vectors and scale by sep/2.
    sep = sep_coeff*pradius
    p0_center = (3*sep/(2*np.sqrt(6)))*np.array([np.sqrt(8/9),0,-1/3])
    p1_center = (3*sep/(2*np.sqrt(6)))*np.array([-np.sqrt(2/9),np.sqrt(2/3),-1/3])
    p2_center = (3*sep/(2*np.sqrt(6)))*np.array([-np.sqrt(2/9),-np.sqrt(2/3),-1/3])
    p3_center = (3*sep/(2*np.sqrt(6)))*np.array([0,0,1])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg


# Generate a 4-particle rod-shaped aggregate. Default orientation is along the z-axis.
def make_rod(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=5, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    # Placement of particles is straightforward. agg radius is 1.5*sep + pradius.
    if mass > 0 or radius > 0: 
        pmass = mass/4
        pradius = radius*(1 + 3*sep_coeff/2)**(-1)

    sep = sep_coeff*pradius
    p0_center = np.array([0,0,-3*sep/2])
    p1_center = np.array([0,0,-sep/2])
    p2_center = np.array([0,0,+sep/2])
    p3_center = np.array([0,0,+3*sep/2])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg

# Generate an 8-particle cube. Centered at the origin and faces parallel to x-, y-, and z- axes.
def make_cube(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=7, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    # agg radius is sqrt(3)*sep/2 + pradius.
    if mass > 0 or radius > 0: 
        pmass = mass/8
        pradius = radius*(1 + np.sqrt(3)*sep_coeff/2)**(-1)

    sep = sep_coeff*pradius
    p0_center = np.array([+sep/2,+sep/2,-sep/2])
    p1_center = np.array([+sep/2,-sep/2,-sep/2])
    p2_center = np.array([-sep/2,-sep/2,-sep/2])
    p3_center = np.array([-sep/2,+sep/2,-sep/2])
    p4_center = np.array([+sep/2,+sep/2,+sep/2])
    p5_center = np.array([+sep/2,-sep/2,+sep/2])
    p6_center = np.array([-sep/2,-sep/2,+sep/2])
    p7_center = np.array([-sep/2,+sep/2,+sep/2])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p4 = Particle(iOrder + 4, iOrgIdx, pmass, pradius, p4_center[0], p4_center[1], p4_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p5 = Particle(iOrder + 5, iOrgIdx, pmass, pradius, p5_center[0], p5_center[1], p5_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p6 = Particle(iOrder + 6, iOrgIdx, pmass, pradius, p6_center[0], p6_center[1], p6_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p7 = Particle(iOrder + 7, iOrgIdx, pmass, pradius, p7_center[0], p7_center[1], p7_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, p4, p5, p6, p7, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
