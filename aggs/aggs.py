#####################################################################
# aggs.py
# Joseph V. DeMartini and Julian C. Marohnic
# Created 10/18/23
# Adapted from earlier work (c. 2022-23) by JCM and JVD
#
# A set of useful methods for aggregates and functions for generating
# aggregate particles, including the regular aggregate shapes:
# dumbbell, diamond, rod, tetrahedron, and cube.
#####################################################################

import numpy as np
from ..particle import Particle
from ..assembly import Assembly
from .. import util

try:
    import scipy.optimize as sco
    from scipy.spatial import ConvexHull
except ModuleNotFoundError:
    print("SciPy module not found. Try installing with 'pip install scipy'")

try:
    from sklearn.decomposition import PCA
    gen_poly = True
except ModuleNotFoundError:
    print("Scikit-learn module not found. Try installing with "
           "'pip install -U scikit-learn'")

try:
    import alphashape
    alphashape_installed = True
except ModuleNotFoundError:
    alphashape_installed = False
    print("Alphashape module not found. Try installing with "
           "'pip install alphashape'")

### Assembly Aggs Functions ###
def agg_max(self): # For these first 3 funcs, shouldn't we do agg_list first then return the values?
    return min([particle.iOrgIdx for particle in self])
Assembly.agg_max = agg_max

def agg_min(self):
    return max([particle.iOrgIdx for particle in self if particle.iOrgIdx < 0])
Assembly.agg_min = agg_min

def agg_range(self):
    agg_tags = [particle.iOrgIdx for particle in self if particle.iOrgIdx < 0]
    return (max(agg_tags), min(agg_tags))
Assembly.agg_range = agg_range

"""
Sample functions using agg_list() instead, for simplicity (perhaps not efficiency?)

def agg_max(self):
    return self.agg_list()[-1]	# or max(self.agg_list())

def agg_min(self):
    return self.agg_list()[0]	# or min(self.agg_list())

def agg_range(self):
    return (self.agg_max(), self.agg_min())
"""

def agg_list(self):
    agg_list = np.unique([particle.iOrgIdx for particle in self if particle.iOrgIdx < 0])

    # Determine whether the assembly contains any aggregates.
    if len(agg_list) == 0:
        print("No aggs in this assembly.")
        return None
    else:	# Return reversed agg_list, starting with -1
        return agg_list[::-1]
Assembly.agg_list = agg_list

def N_aggs(self):
    # Return number of aggs in the assembly.
    return len(self.agg_list())
Assembly.N_aggs = N_aggs

def get_agg(self, iOrgIdx):
    # Returns a new assembly consisting only of particles in the desired aggregate.
    if not isinstance(iOrgIdx, int):
        raise TypeError("Warning: get_agg() takes a single negative integer as its argument.")
    if iOrgIdx >= 0:
        raise ValueError("Warning: get_agg() takes a single negative integer as its argument.")

    matches = [particle for particle in self if particle.iOrgIdx == iOrgIdx]
    return Assembly(*matches, units=self.units, time=self.time)
Assembly.get_agg = get_agg

def del_aggs(self, *iOrgIdxs):
    # Delete specified aggs from the assembly.
    for element in iOrgIdxs:
        if not isinstance(element, int):
            raise TypeError("Warning: del_agg() takes negative integers as arguments.\n"
                            "If you would like to use a list to specify the aggs to be deleted, "
                            "use the '*' operator.\n")
        if element >= 0:
            raise ValueError("Warning: del_agg() takes negative integers as arguments.\n"
                            "If you would like to use a list to specify the aggs to be deleted, "
                            "use the '*' operator.\n")
    del_list = [particle.iOrder for particle in self if particle.iOrgIdx in iOrgIdxs]
    self.del_particles(*del_list)
Assembly.del_aggs = del_aggs

def pop_agg(self, iOrgIdx):
    # "Pop" the desired agg from the assembly, deleting it from the assembly and returning
    #   a new copy.
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

def fix_orphans(self):
    # Find any single particles with iOrgIdx < 0 ("orphans") and set iOrgIdx = iOrder.
    # **Currently very slow. Consider ways to make this operation more efficient. **
    agg_tags = [particle.iOrgIdx for particle in self if particle.iOrgIdx < 0]
    orphans = []
    for index in agg_tags:	# I believe this is the slow part.
        if agg_tags.count(index) == 1:	#Here you double search the whole size of your array
            orphans.append(index)

    if len(orphans) == 0:
        print("No orphan particles found in this assembly.")
        return

    for particle in self:
        if particle.iOrgIdx in orphans:
            particle.iOrgIdx = particle.iOrder

    print(len(orphans), "orphan(s) corrected.")
Assembly.fix_orphans = fix_orphans

def condense_aggs(self, direction='d'):
    # Renumbers negative iOrgIdxs consecutively, keeping aggregates together
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

def all_aggs(self, time=0.0, units=None):
    # Return a list of assemblies, each containing one agg from the assembly passed in.
    # Units match those of the assembly argument unless an alternative is specified.
    if units == None:
        units = self.units
    # Check for valid specification of units.
    if units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")
    # Check for valid time specification
    if not isinstance(time, float):
        raise TypeError("Time value must be a positive, real number.")
    if time < 0.0:
        raise ValueError("Time value must be a positive, real number.")

    # Get all negative iOrgIdx values.
    agg_list = self.agg_list()

    # Create a new assembly for each agg and add it to the list.
    aggs = []
    for i in agg_list:
        aggs.append(Assembly(*[particle for particle in self if particle.iOrgIdx == i],
                     time=time, units=units))

    return aggs
Assembly.all_aggs = all_aggs


### Internal Functions for Regular Aggs ###
def _checkRegularAggMR(mass, radius, pmass, pradius):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius or pmass and pradius must "
                           "both be positive and non-zero.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("User can specify *either* aggregate mass and radius *or* "
                          "constituent sphere mass and radius, but not both.")


def _rotateAgg(agg, orient_init, orient_final):
    axis = np.cross(orient_init, orient_final)
    angle = util.angle_between(orient_init, orient_final)

    if np.linalg.norm(axis) == 0 or angle == 0:
        return agg
    else:
        unit_axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        return agg


def _genRegularAgg(iOrder, iOrgIdx, center, orientation, color, pmass, pradius, sep,
                    pcenter_array, time, units):
    # Make constituent particles
    pcenters = sep * pcenter_array

    particle_list = []
    for i in range(len(pcenters)):
        particle_list.append(Particle(iOrder + i, iOrgIdx, pmass, pradius,
                              pcenters[i, 0], pcenters[i, 1], pcenters[i, 2],
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=units))

    # Create aggregate
    agg = Assembly(*particle_list, time=time, units=units)

    # Rotate & translate aggregate to user-specified orientation and position
    agg = _rotateAgg(agg, np.array([0, 0, 1]), np.array(orientation))
    agg.set_center(center)

    return agg


### Generating Regular Aggs ###
def makeCube(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1),
              color=7, pmass=0, pradius=0, sep_coeff=np.sqrt(3), time=0.0, units='pkd'):
    # Input checking
    _checkRegularAggMR(mass, radius, pmass, pradius)

    # Define constants from user inputs
    if mass > 0 or radius > 0:	# Agg radius = sqrt(3)*sep/2 + pradius
        pmass = mass/8
        pradius = radius/(1 + np.sqrt(3) * sep_coeff/2)

    sep = sep_coeff * pradius

    # Place constituent particles
    particle_centers = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                                [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])

    # Create aggregate
    cube = _genRegularAgg(iOrder, iOrgIdx, center, orientation, color, pmass, pradius,
                           sep/2, particle_centers, time, units)
    return cube


def makeDB(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1),
              color=2, pmass=0, pradius=0, sep_coeff=np.sqrt(3), time=0.0, units='pkd'):
    # Input checking
    _checkRegularAggMR(mass, radius, pmass, pradius)

    # Define constants from user inputs
    if mass > 0 or radius > 0:
        pmass = mass/2
        pradius = radius/(1 + sep_coeff/2)

    sep = sep_coeff * pradius

    # Place constituent particles
    particle_centers = np.array([[0, 0, -1], [0, 0, 1]])

    # Create aggregate
    dumbbell = _genRegularAgg(iOrder, iOrgIdx, center, orientation, color, pmass, pradius,
                           sep/2, particle_centers, time, units)
    return dumbbell


def makeDiamond(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1),
              color=12, pmass=0, pradius=0, sep_coeff=np.sqrt(3), time=0.0, units='pkd'):
    # Input checking
    _checkRegularAggMR(mass, radius, pmass, pradius)

    # Define constants from user inputs
    if mass > 0 or radius > 0:
        pmass = mass/4
        pradius = radius/(1 + sep_coeff)

    sep = sep_coeff * pradius

    # Place constituent particles
    particle_centers = np.array([[0, 0, -1], [0, 0, 1], [-0.5, 0, 0], [0.5, 0, 0]])

    # Create aggregate
    diamond = _genRegularAgg(iOrder, iOrgIdx, center, orientation, color, pmass, pradius,
                           sep, particle_centers, time, units)
    return diamond


def makeRod(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1),
              color=5, pmass=0, pradius=0, sep_coeff=np.sqrt(3), time=0.0, units='pkd'):
    # Input checking
    _checkRegularAggMR(mass, radius, pmass, pradius)

    # Define constants from user inputs
    if mass > 0 or radius > 0:
        pmass = mass/4
        pradius = radius/(1 + 3*sep_coeff/2)

    sep = sep_coeff * pradius

    # Place constituent particles
    particle_centers = np.array([[0, 0, -3], [0, 0, -1], [0, 0, 1], [0, 0, 3]])

    # Create aggregate
    rod = _genRegularAgg(iOrder, iOrgIdx, center, orientation, color, pmass, pradius,
                           sep/2, particle_centers, time, units)
    return rod


def makeTetra(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1),
              color=3, pmass=0, pradius=0, sep_coeff=np.sqrt(3), time=0.0, units='pkd'):
    # Input checking
    _checkRegularAggMR(mass, radius, pmass, pradius)

    # Define constants from user inputs
    if mass > 0 or radius > 0:
        pmass = mass/4
        pradius = radius/(1 + sep_coeff/2)

    sep = sep_coeff * pradius

    # Place constituent particles
    particle_centers = np.array([[np.sqrt(8/9), 0, -1/3], [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
                                  [-np.sqrt(2/9), -np.sqrt(2/3), -1/3], [0, 0, 1]])

    # Create aggregate
    tetrahedron = _genRegularAgg(iOrder, iOrgIdx, center, orientation, color, pmass, pradius,
                           3*sep/(2*np.sqrt(6)), particle_centers, time, units)
    return tetrahedron

### Generating Irregular Aggs ###

## Generate a random convex/concave polyhedron
# Reference: https://link-springer-com.proxy-um.researchport.umd.edu/content/pdf/10.1007/s12205-018-0182-8.pdf
def _genPointCloud(N_points):
    if N_points <= 3:
        raise ValueError("Cannot generate an irregular polyhedron with fewer than 4 vertices")
    rng = np.random.default_rng()
    points = 2.0 * rng.random(size=[int(N_points), 3]) - 1.0
    return points


def _PCARotation(point_cloud):	#This choice of method is thanks to a chatGPT suggestion
    # Principal component analysis (PCA) to align principal axes with cartesian axes
    pca = PCA(n_components=3)
    pca.fit(point_cloud)
    rot_matrix = pca.components_.T
    new_points = rot_matrix.dot(point_cloud.T).T
    return rot_matrix, new_points

def _minBBox(vertices):
    # Define the minimum bounding box of a hull
    dimensions = np.empty(3)
    center = np.empty(3)
    for i in range(3):	# Iterate over length, width, height (x,y,z)
        min, max = np.min(vertices[:, i]), np.max(vertices[:, i])
        dimensions[i] = max - min

    return dimensions

def _pointExtension(points, alphas):
    # Extend all points to better fit the desired dimensions
    return points + points*alphas

def _genPolyHull(Radius, axis_ratios, N_points, concave=False):
    # Generate a random polyhedral with the given axis ratios and
    #  a number of vertices similar to N_points (w/in an order of magnitude)

    # Initialize desired component axis lengths
    input_dims = 2*Radius * np.append(1.0, axis_ratios)

    # Generate a random cloud of points
    init_points = _genPointCloud(N_points)

    # Rotate the point cloud so that the minimum bounding box will align w/ Cartesian axes
    rot_matrix, points_rotated = _PCARotation(init_points)

    # Create the convex hull around the point cloud
    if concave:
        init_hull = alphashape.alphashape(points_rotated, alpha=0.1*Radius)	# 0.1 is a choice!
        hull_vertices = init_hull.vertices
    else:
        init_hull = ConvexHull(points_rotated)
        hull_vertices = init_hull.points[init_hull.vertices]

    bbox_dims = _minBBox(hull_vertices)
    while not np.array_equal(input_dims, bbox_dims):
        # Adjust point locations to maximally fill volume specified by input radius, axis ratios
        extension_coeffs = 1 - (bbox_dims/input_dims)
        hull_vertices = _pointExtension(hull_vertices, extension_coeffs)

        # Determine minimum bounding box around the hull
        bbox_dims = _minBBox(hull_vertices)

    # Calculate hull around final constrained points
    if concave:
        final_hull = alphashape.alphashape(hull_vertices, alpha=0.1*Radius)
    else:
        final_hull = ConvexHull(hull_vertices)

    return final_hull

## Fill a polyhedral hull

def _genInitialSpheres(radius, vertices, ndivisions):
    # Generate a grid of points within the bounding box of the hull
    #  Initial radii set to Rmin
    init = np.empty([3,ndivisions])
    for i in range(3):
        min, max = np.min(vertices[:,i]), np.max(vertices[:,i])
        di = (max - min)/ndivisions
        init[i] = np.fromiter((min + (j * di) for j in range(ndivisions)), float)

    X, Y, Z = np.meshgrid(*init)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    return points, radius * np.ones(points.shape[-1])

def _genLinearConstraintConvex(hull):
    # Define plane equations of facets. Qhull hyperplane equations given s.t. Ax <= -b
    #  hull.equations returns [x,y,z,b], A = [x,y,z]
    normals = hull.equations[:,:-1]

    A = np.hstack([normals, np.ones([len(normals),1])])
    B = -hull.equations[:,-1]

    return sco.LinearConstraint(A, lb=-np.inf, ub=B)

def _genLinearConstraintConcave(hull):	#See above for descriptions
    A = np.hstack([hull.facets_normal, np.ones([len(hull.facets_normal),1])])
    B = np.einsum('ij,ij->i', hull.facets_normal, hull.facets_origin)

    return sco.LinearConstraint(A, lb=-np.inf, ub=B)

def _genNonlinearConstraint(spheres, ovlp):
    # C(X) <= 0 (in our case)
    function = lambda x: -(x[0] - spheres[0])**2 - (x[1] - spheres[1])**2 - (x[2] - spheres[2])**2 + ((1 - ovlp) * x[3] + spheres[3])**2
    return sco.NonlinearConstraint(function, lb=-np.inf, ub=0)

def _fillPoly(hull, r_min, r_max, overlap_fraction, concave=False):
    # Fill a polygon with spheres. Follows Sec 4 of the above referenced paper.

    # Generate initial values and radii
    if concave:
        hull_vertices = hull.vertices
    else:
        hull_vertices = hull.points[hull.vertices]

    sph_pos, sph_radii = _genInitialSpheres(r_min, hull_vertices, 10)	# 10 is a choice, to get enough total spheres
    spheres_init = np.vstack([sph_pos, sph_radii]).T

    spheres_final = np.empty([4,0])	# Output list for particle positions & radii

    ## Nonlinear optimization equation: minimize F(X) = -Radius s.t.
    ##  constraints: A*X<=B, A_eq*X=B_eq, (linear); AND C(X)<=0, C_eq(X)=0 (nonlinear)
    min_func = lambda x: -x[3]	# Maximize sphere radius to fit inside the hull

    # Transform polyhedron face to the form of point + outward normal; define linear constraint
    if concave:
        linear_constraint = _genLinearConstraintConcave(hull)
    else:
        linear_constraint = _genLinearConstraintConvex(hull)

    # Set bounds for particle x, y, z, and r (necessary?)
    bounds = []
    for i in range(3):
        bounds += [(np.min(hull_vertices[:,i]), np.max(hull_vertices[:,i]))]
    bounds += [(r_min, r_max)]

    # Select a sphere position from the "initial" point set as the "initial guess"
    for sph in spheres_init:
        spheres_final = np.hstack([spheres_final, sph[:,np.newaxis]])
        cons = (linear_constraint, _genNonlinearConstraint(spheres_final, overlap_fraction))

        # Find likely sph position w/ scipy.opt.minimize()
        result = sco.minimize(min_func, sph, method='SLSQP', bounds=bounds, constraints=cons)

        # If sph radius outside limiting radius, discard
        if result.success:
            spheres_final = np.hstack((spheres_final.T[:-1].T, result.x[:,np.newaxis]))
        else:
            spheres_final = spheres_final.T[:-1].T

    return spheres_final

## Generate and fill an irregular shape ##
def _sphVolume(radii):
    # Volume of a sphere
    return 4 * np.pi * (radii**3) / 3

def _calcMassFromTotal(m_tot, radii):
    # Calculate sphere mass from agg total mass. Assumes equal density spheres.
    volumes = _sphVolume(radii)
    volume_fractions = volumes/volumes.sum()
    return m_tot * volume_fractions

def _calcMassFromDens(density, radii):
    # Caluclate sphere mass from agg density. Assumes equal density spheres.
    volumes = _sphVolume(radii)
    return density * volumes

def _genIrregularAgg(iOrder, iOrgIdx, center, orientation, color, pmasses, pradii,
                      pcenter_array, time, units):
    # Make constituent particles
    particle_list = []
    for i in range(len(pcenter_array)):
        particle_list.append(Particle(iOrder + i, iOrgIdx, pmasses[i], pradii[i],
                              pcenter_array[i, 0], pcenter_array[i, 1], pcenter_array[i, 2],
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=units))

    # Create aggregate
    agg = Assembly(*particle_list, time=time, units=units)

    # Rotate and translate aggregate to user-specified orientation and position
    agg = _rotateAgg(agg, np.array([0, 0, 1]), np.array(orientation))
    agg.set_center(center)

    return agg

def makeIrregPoly(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1),
                  color=10, pdens=0, pradius_min=0, pradius_max=0, sep_coeff=np.sqrt(3),
                  axis_ratios=(0.75,0.5), concave=False, time=0.0, units='pkd'):
    # Generate a polyhedral hull and fill it efficienctly with spheres in the size range
    # [pradius_min, pradius_max]
    if (radius <= 0 or pradius_min <= 0) and (mass <= 0 or pdens_min <= 0):
        raise ValueError("User must specify both radius and minimum constituent particle "
                          "radius, plus either the total agg mass or the density of "
                          "the constituent particles. All of these quantities must be "
                          "positive.")
    if (mass > 0 and pdens > 0):
        raise ValueError("User can specify *either* aggregate mass *or* constituent "
                          "sphere density, but not both.")
    if (pradius_max < pradius_min):
        raise ValueError("Maximum allowable particle radius must be greater than or "
                          "equal to minimum allowable constituent particle radius.")
    #if (concave and not alphashape_installed):
    #    raise ValueError("User cannot specify concave polyhedra without downloading "
    #                      "and installing the 'alphashape' library from PyPI.")

    # Generate polyhedral hull based on input parameters
    #  -Should make some choice about nVertices? (3rd argument in _genPolyHull)
    poly_hull = _genPolyHull(radius, axis_ratios, 10*np.floor(radius/pradius_min), concave=concave)

    # Find positions [0-2] and radii [3] of particles to most efficiently fill the agg
    beta_ovlp = (sep_coeff/2) - 1
    constituent_array = _fillPoly(poly_hull, pradius_min, pradius_max, beta_ovlp, concave=concave)
    particle_centers = constituent_array[:3].T
    pradii = constituent_array[3]

    # Calculate masses based on input mass or particle density
    if mass > 0:
        pmasses = _calcMassFromTotal(mass, pradii)
    elif pdens > 0:
        pmasses = _calcMassFromDens(pdens, pradii)

    # Create aggregate
    polyhedron = _genIrregularAgg(iOrder, iOrgIdx, center, orientation, color, pmasses,
                            pradii, particle_centers, time, units)

    return polyhedron
