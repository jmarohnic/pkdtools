##################################################################
# io.py
# Julian C. Marohnic
# Created: 8/16/23
# 
# Read and write functions for pkdtools. Both rely on ssio, which
# handles reading and writing from the binary ss format. ss_in() 
# and ss_out() mediate between the ssio functions and the Particle
# and Assembly class structures of pkdtools. Adapted from ssedit.
##################################################################

import numpy as np

from .particle import Particle
from .assembly import Assembly
from . import ssio

def ss_in(filename, units='pkd'):
    try:
        header, ssdata = ssio.read_SS(filename, 'y')
    except:
        print("Error: Invalid ss file.")
        return 1

    new_assembly = Assembly(time=header[0])
    for i in range(header[1]):	# JVD concision edit made here
        new_assembly.add_particles(Particle(*ssdata[:,i]))

    # Set units if needed. Default is pkd, from Assembly().
    if units != 'pkd':
        new_assembly.units = units

    return new_assembly

def ss_out(assembly, filename):
    if not isinstance(assembly, Assembly):
        raise TypeError("Only assemblies can be written to ss files.")

    # Make sure we are writing real particles. 
    bad_list = []
    for i, element in enumerate(assembly):
        if not isinstance(element, Particle):
            bad_list.append(i)

    if bad_list != []:
        raise TypeError("Can only add particles to assemblies. The following assembly element(s) are not proper particles:\n"
                        f"{bad_list}.")

    # Warn user about any duplicate and non-sequential iOrder fields. ssio.py
    # will always renumber particles as 0,1,2,... Need to reorder to do this.
    # Don't want to reorder the actual assembly before writing, so need to make a copy. 
    ss_copy = assembly.copy()
    ss_copy.sort_iOrder()

    dup_list = []
    seq_warn = 0

    for i in range(1, ss_copy.N()):
        if ss_copy[i].iOrder == ss_copy[i-1].iOrder:
            dup_list.append(ss_copy[i].iOrder)
        if ss_copy[i].iOrder != i:
            seq_warn = 1

    if dup_list != []:
        print("Warning: the following iOrder numbers appear *at least* twice in the assembly to be written:\n" 
              f"{list(set(dup_list))}.")

    if seq_warn == 1:
        print("Warning: the iOrder values of the particles you are trying to write are not\n"
              "in sequential, increasing order beginning with '0'. This numbering will not\n"
              "be respected by ssio.py. You may call the 'condense()' method on your assembly\n"
              "to see how your particles will be renumbered.")

    # Ensure that units are 'pkd' before writing.
    ss_copy.units = 'pkd'

    # Pack up data for writing with ssio.
    iOrder_list = [particle.iOrder for particle in ss_copy]
    iOrgIdx_list = [particle.iOrgIdx for particle in ss_copy]
    m_list = [particle.m for particle in ss_copy]
    R_list = [particle.R for particle in ss_copy]
    x_list = [particle.x for particle in ss_copy]
    y_list = [particle.y for particle in ss_copy]
    z_list = [particle.z for particle in ss_copy]
    vx_list = [particle.vx for particle in ss_copy]
    vy_list = [particle.vy for particle in ss_copy]
    vz_list = [particle.vz for particle in ss_copy]
    wx_list = [particle.wx for particle in ss_copy]
    wy_list = [particle.wy for particle in ss_copy]
    wz_list = [particle.wz for particle in ss_copy]
    color_list = [particle.color for particle in ss_copy]
    """
    # JVD Suggested edit to save on number of loops:
    iOrder_list, iOrgIdx_list, ... , color_list = ([] for j in range(14))
    for particle in ss_copy:
        iOrder_list.append(particle.iOrder)
        ...
        color_list.append(particle.color)
    # This method only loops once through all particles. More efficient for large N.
    """

    new_ss = np.array([iOrder_list, iOrgIdx_list, m_list, R_list, \
                       x_list, y_list, z_list, vx_list, vy_list, vz_list, \
                       wx_list, wy_list, wz_list, color_list])

    try:
        ssio.write_SS(new_ss, filename, time=ss_copy.time)	# JVD EDIT
    except:
        print("Error: Write to ss file failed.")
        return 1
