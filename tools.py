##################################################################
# tools.py
# Julian C. Marohnic
# Created: 8/23/23 
#
# Miscellaneous functions for manipulating particles and assemblies.
# Also includes the handy viz() function.
##################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .assembly import Assembly
from . import util

# Combine two existing assemblies into one new assembly. User may supply units, otherwise new assembly 
# will default to pkd units. Similar to other functions/methods for manipulating assemblies, join()
# will create a new assembly composed of copies of the input assemblies, and further manipulations
# will not affect the original inputs.
def join(*assemblies, units=None, time=0.0):
    # Check for valid specification of units.
    if units != None and units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")
    if not isinstance(time, float) and not isinstance(time, int):
        raise TypeError("Assembly time must be a non-negative number.")
    if time < 0:
        raise ValueError("Assembly time must be a non-negative number.")
    for element in assemblies:
        if not isinstance(element, Assembly):
            raise ValueError("Can only join assemblies.")

    # Default setting
    if units == None:
        units = 'pkd'

    new = Assembly(time=time)

    for element in assemblies:
        new.add_particles(*element)

    # Make sure time and all added particles conform to specified units before returning.
    new.units = units
    return new

# Similar to the del_particles() method, but returns a new
# assembly with *only* the input iOrders.
def subasbly(assembly, *iOrders, units='None', time=0.0):	# JVD time EDIT
    sub = Assembly(*[particle.copy() for particle in assembly if particle.iOrder in iOrders])

    if time:
        subm.time = time
    else:
        sub.time = assembly.time

    if units != 'None':
        sub.units = units
    else:
        sub.units = assembly.units

    return sub

# Visualize an assembly of particles on the fly. Resolution determines how round or blocky the 
# particles appear. Higher res takes longer to render. Large numbers of particles won't work well here.
# Default value is 10, so manually setting a higher value will take longer to render and vice versa
# when setting a lower value.
def viz(assembly, resolution=10):
    if not isinstance(assembly, Assembly):
        raise TypeError("Only assemblies can be visualized.")
    if assembly.N() < 2:
        raise ValueError("Only assemblies with 2 or more particles may be visualized.")

    xlist = []
    ylist = []
    zlist = []
    radlist = []
    colorlist = []

    for particle in assembly:
        xlist.append(particle.x)
        ylist.append(particle.y)
        zlist.append(particle.z)
        radlist.append(particle.R)
        colorlist.append(util.color_translate(particle.color))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Hack to get square projection since matplotlib somehow does not support this currently. Find the center point of the assembly
    # between the most extreme particles in x, y, and z. This should be the origin of the plot. Will cause problems when there are particles
    # separated by large distances, but this should be fine for now.
    centx = assembly.xbounds()[0] + (assembly.xbounds()[1] - assembly.xbounds()[0])/2
    centy = assembly.ybounds()[0] + (assembly.ybounds()[1] - assembly.ybounds()[0])/2
    centz = assembly.zbounds()[0] + (assembly.zbounds()[1] - assembly.zbounds()[0])/2
    cent = [centx, centy, centz]
    # Would be nice to scale each axis individually, but this will lead to weirdly stretched
    # particles. matplotlib currently does not support setting a 'square' aspect ratio in 3D.
    max_dist = assembly.R()
    max_rad = max(radlist)
    
    for x, y, z, rad, color in zip(xlist, ylist, zlist, radlist, colorlist):
        X, Y, Z = util.makesphere(x, y, z, rad, resolution=resolution)
        ax.plot_surface(X, Y, Z, color=color)

    ax.axes.set_xlim3d(left=cent[0]-max_dist-max_rad, right=cent[0]+max_dist+max_rad)
    ax.axes.set_ylim3d(bottom=cent[1]-max_dist-max_rad, top=cent[1]+max_dist+max_rad)
    ax.axes.set_zlim3d(bottom=cent[2]-max_dist-max_rad, top=cent[2]+max_dist+max_rad)
    ax.set_xlabel(f'x ({assembly.units})')
    ax.set_ylabel(f'y ({assembly.units})')
    ax.set_zlabel(f'z ({assembly.units})')

    plt.show()
