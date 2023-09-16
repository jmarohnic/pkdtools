##################################################################
# particle.py
# Julian C. Marohnic
# Created: 8/16/23
#
# A dedicated "Particle" class for pkdtools. Includes all standard 
# pkdgrav particle parameters: iOrder, iOrgIdx, mass, radius, x-, y-,
# and z- components of positions, velocities, and spins, and color.
# Adapted from ssedit (2022), including work by JCM and JVD.
##################################################################

import numpy as np

from . import util

# Particle data structure. Attributes correspond to standard pkdgrav particle parameters.
class Particle:
    def __init__(self, iOrder, iOrgIdx, m, R, x=0.0, y=0.0, z=0.0,
                 vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0, color=3,
                 units='pkd'):
        self.iOrder = iOrder
        self.iOrgIdx = iOrgIdx
        self.m = m
        self.R = R
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.color = color
        self._units = units # Need to initialize like this to avoid bugs with the units setter function.
        self.units = units

    # Input checking for attributes where this makes sense.
    @property
    def iOrder(self):
        return self._iOrder

    @iOrder.setter
    def iOrder(self, value):
        if not isinstance(value, int):
            raise TypeError("iOrder must be a non-negative integer.")
        if value < 0:
            raise ValueError("iOrder must be a non-negative integer.")
        self._iOrder = value

    @property
    def iOrgIdx(self):
        return self._iOrgIdx

    @iOrgIdx.setter
    def iOrgIdx(self, value):
        if not isinstance(value, int):
            raise TypeError("iOrgIdx must be an integer.")
        self._iOrgIdx = value

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("Particle mass must be a positive number.")
        if value <= 0:
            raise ValueError("Particle mass must be a positive number.")
        self._m = float(value)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("Particle radius must be a positive number.")
        if value <= 0:
            raise ValueError("Particle radius must be a positive number.")
        self._R = float(value)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if not isinstance(value, int):
            raise TypeError("Particle color must be an integer between -N and 255.")
        #if value < 0 or value > 255:
            #raise ValueError("Particle color must be an integer between 0 and 255.")
        self._color = value

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value not in ['pkd', 'cgs', 'mks']:
            raise ValueError("Particle units must be one of 'pkd', 'cgs', or 'mks'. Default is 'pkd'.") 
        self.convert(value)

    # Return particle position vector. Can come in handy.
    def pos(self):
        return np.array([self.x, self.y, self.z])

    # Return particle velocity vector.
    def vel(self):
        return np.array([self.vx, self.vy, self.vz])

    # Return particle spin vector.
    def spin(self):
        return np.array([self.wx, self.wy, self.wz])

    # Set particle position with a vector input.
    def set_pos(self, pos, units=None):
        if not isinstance(pos, np.ndarray) and not isinstance(pos, tuple) and not isinstance(pos, list):
            raise TypeError("Input position must be a 3-element vector.")
        if len(pos) != 3:
            raise ValueError("Input position must be a 3-element vector.")
        # If units are supplied, use them, otherwise default to particle's current units.
        if units != None:
            self.units = units

        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    # Set particle velocity with a vector input.
    def set_vel(self, vel, units=None):
        if not isinstance(vel, np.ndarray) and not isinstance(vel, tuple) and not isinstance(vel, list):
            raise TypeError("Input velocity must be a 3-element vector.")
        if len(vel) != 3:
            raise ValueError("Input velocity must be a 3-element vector.")
        if units != None:
            self.units = units

        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]

    # Set particle spin with a vector input.
    def set_spin(self, w, units=None):
        if not isinstance(w, np.ndarray) and not isinstance(w, tuple) and not isinstance(w, list):
            raise TypeError("Input spin must be a 3-element vector.")
        if len(w) != 3:
            raise ValueError("Input spin must be a 3-element vector.")
        if units != None:
            self.units = units

        self.wx = w[0]
        self.wy = w[1]
        self.wz = w[2]

    # Create a new particle with the same parameter values.
    def copy(self):
        return Particle(self.iOrder, self.iOrgIdx, self.m, self.R, self.x, self.y, self.z,
                 self.vx, self.vy, self.vz, self.wx, self.wy, self.wz, self.color, self.units)

    # Convert particle units. New particle units assumed to be 'pkd' unless specified.
    def convert(self, value='pkd'):
        if value == 'pkd':
            if self.units == 'pkd':
                return None
            elif self.units == 'cgs':
                util.cgs2pkd(self)
                return None
            elif self.units == 'mks':
                util.mks2pkd(self)
                return None
        elif value == 'cgs':
            if self.units == 'cgs':
                return None
            elif self.units == 'pkd':
                util.pkd2cgs(self)
                return None
            elif self.units == 'mks':
                util.mks2cgs(self)
                return None
        elif value == 'mks':
            if self.units == 'mks':
                return None
            elif self.units == 'pkd':
                util.pkd2mks(self)
                return None
            elif self.units == 'cgs':
                util.cgs2mks(self)
                return None
        else:
            raise ValueError("Something has gone wrong here!")
            return 1

    # Display particle attributes when printed.
    def __str__(self):
        return (f'[{self.iOrder}, {self.iOrgIdx}, {self.m}, {self.R}, '
                f'{self.x}, {self.y}, {self.z}, {self.vx}, {self.vy}, {self.vz}, '
                f'{self.wx}, {self.wy}, {self.wz}, {self.color}, {self.units}]')
