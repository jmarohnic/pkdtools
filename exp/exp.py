##################################################################
# exp.py
# Julian C. Marohnic
# Created: 8/17/23
#
# "Experimental" module for pkdtools. Home for any miscellaneous
# functions not intended or ready for general use.
##################################################################

import numpy as np

from .particle import Particle
from .assembly import Assembly

# Embed a spherical boulder in a rubble pile. Center argument is defined
# relative to agg COM.
def embed_boulder(assembly, center, radius, units='pkd'):
    COM = assembly.com()
    agg_max = min([particle.iOrgIdx for particle in assembly]) # Most negative iOrgIdx is last agg.
    boulder_list = []
    for particle in assembly:
        if np.linalg.norm(particle.pos() - (COM + center)) <= radius:
            boulder_list.append(particle.iOrgIdx)

    for particle in assembly:
        if particle.iOrgIdx in boulder_list:
            particle.iOrgIdx = agg_max - 1
            particle.color = 1
