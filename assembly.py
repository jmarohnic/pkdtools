##################################################################
# assembly.py
# Julian C. Marohnic
# Created: 8/16/23
#
# The "Assembly" class and base methods. Imported by default with 
# pkdtools. An assembly is fundamentally a list of particles, with
# additional units and time fields. The basic format for groups of
# particles in pkdtools. Adapted from ssedit.py (2022), including 
# work by JCM and JVD.
##################################################################

import numpy as np
import scipy.spatial as ss

from .particle import Particle
from . import util

# "Assembly" data structure. Holds 1 or more particles. This could maybe be implemented in a cleaner way.
# NOTE: Must unpack list argument to Assembly(). E.g.: Assembly(*<list of particles>), NOT Assembly(<list of particles>)
class Assembly(list):
    def __init__(self, *particles, units='pkd', time=0.0):
        # Set timestamp for the assembly. Can't use the setter for the initialization since we haven't established units yet.
        if not isinstance(time, float):
            raise TypeError("Time value must be a positive, real number.")
        if time < 0.0:
            raise ValueError("Time value must be a positive, real number.")
        #if len(particles) < 1:
        #    raise ValueError("An assembly must be initialized with at least one particle.")
        self._time = time
        # Establish units for the assembly, using the "property" approach below to verify input. 'pkd' is default.
        self._units = units
        self.units = units
        # We initialize assembly with a copy of each input particle to keep the assembly indepenent from the particles
        # it's made from.
        to_add = []
        for element in particles:
            if not isinstance(element, Particle):
                raise TypeError("All assembly elements must be particles.")
            p_copy = element.copy()
            # Make sure each particle is using units consistent with the assembly choice.
            if p_copy.units != self.units:
                p_copy.convert(units)

            to_add.append(p_copy)

        list.__init__(self, to_add)

    # Manage units for the assembly. Whenever changing, check that choice is valid and ensure each particle is consistent.
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value not in ['pkd', 'cgs', 'mks']:
            raise TypeError("Assembly units must be one of 'pkd', 'cgs', or 'mks'. Default is 'pkd'.") 
        for particle in self:
            if particle.units != value:
                particle.convert(value)
        self.convert_time(value)
        self._units = value

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        if not isinstance(value, float) and not isinstance (value, int):
            raise TypeError("Assembly time must be a non-negative number.")
        if value < 0:
            raise ValueError("Assembly time must be a non-negative number.")
        self._time = float(value)

    def N(self):
        return len(self)

    def M(self):
        return sum([particle.m for particle in self])

    def convert_time(self, value='pkd'):
        if value == 'pkd':
            if self.units == 'pkd':
                return None
            elif self.units == 'cgs' or self.units =='mks':
                util.sec2pkd(self)
                return None
        if value == 'cgs' or value == 'mks':
            if self.units == 'pkd':
                util.pkd2sec(self)
                return None
            elif self.units == 'cgs' or self.units == 'mks':
                return None
        else:
            raise ValueError("Something has gone wrong here!")
            return 1

    # Return a tuple with the minimum and maximum x positions over all particles in the assembly. There may be a nicer way to package this info?
    # Originally called "xrange" etc., but unfortunately this name was already taken by a Python builtin.
    def xbounds(self):
        allx = [particle.x for particle in self]
        return (min(allx), max(allx))

    def ybounds(self):
        ally = [particle.y for particle in self]
        return (min(ally), max(ally))

    def zbounds(self):
        allz = [particle.z for particle in self]
        return (min(allz), max(allz))

    # Return center of mass position of assembly.
    def com(self):
        pos = 0.0
        for particle in self:
            pos += particle.m*particle.pos()
        return pos/self.M()

    # Return center of mass velocity of assembly.
    def comv(self):
        vel = 0.0
        for particle in self:
            vel += particle.m*particle.vel()
        return vel/self.M()

    # Calculate "center" of an assembly. I.e., the point halfway between the min and max x, y, and z values.
    # Not equivalent to center of mass!
    def center(self):
        xmin, xmax = self.xbounds()
        ymin, ymax = self.ybounds()
        zmin, zmax = self.zbounds()

        centx = xmin + (xmax - xmin)/2
        centy = ymin + (ymax - ymin)/2
        centz = zmin + (zmax - zmin)/2

        return np.array([centx, centy, centz])

    # Calculate angular frequency vector of an assembly.
    def ang_freq(self):
        return np.dot(np.linalg.inv(self.I()), self.L())

    # Calculate rotational frequency vector of an assembly. Probably not necessary to include with ang_freq().
    def freq(self):
        return self.ang_freq()/(2*np.pi)

    # Calculate the net spin period of an assembly.
    def period(self):
        return 1/np.linalg.norm(self.freq())

    # Translate all particles so that the assembly has the new center of mass position.
    # Roundoff error could be an issue here when moving com by a large amount. Should
    # be fine when aggs are relatively small?
    def set_com(self, com, units=None):
        if not isinstance(com, np.ndarray) and not isinstance(com, tuple) and not isinstance(com, list):
            raise TypeError("Input center of mass must be a 3-element vector.")
        if len(com) != 3:
            raise ValueError("Input center of mass must be a 3-element vector.")
        # If units are supplied, use them, otherwise default to assembly's current units.
        if units != None:
            self.units = units

        # Ensure that com is a numpy array and preserve original center of mass location.
        com = np.array(com)
        current_com = self.com()

        for particle in self:
            rel_pos = particle.pos() - current_com
            particle.set_pos(com + rel_pos)

    # Edit all particles so that the assembly has the new center of mass velocity.
    def set_comv(self, comv, units=None):
        if not isinstance(comv, np.ndarray) and not isinstance(comv, tuple) and not isinstance(comv, list):
            raise TypeError("Input velocity must be a 3-element vector.")
        if len(comv) != 3:
            raise ValueError("Input velocity must be a 3-element vector.")
        if units != None:
            self.units = units

        comv = np.array(comv)
        current_comv = self.comv()

        for particle in self:
            rel_vel = particle.vel() - current_comv
            particle.set_vel(comv + rel_vel)

    # Translate assembly to match the desired center location. Occasionally useful, especially 
    # when making figures. Default behavior with no argument is to move center to (0,0,0).
    def set_center(self, center=(0,0,0), units=None):
        if not isinstance(center, np.ndarray) and not isinstance(center, tuple) and not isinstance(center, list):
            raise TypeError("Input velocity must be a 3-element vector.")
        if len(center) != 3:
            raise ValueError("Input velocity must be a 3-element vector.")
        if units != None:
            self.units = units

        # Calculate necessary displacement.
        current_cent = self.center()
        disp = center - current_cent

        for particle in self:
            pos = particle.pos()
            particle.set_pos(pos + disp)

    # Get an estimate of assembly radius. Naive approach here just picks greatest distance
    # between assembly COM and any one particle in the assembly, plus that particle's radius.
    def R(self):
        # If assembly contains only one particle, radius is just that particle's radius.
        if self.N() == 1:
            return self[0].R

        COM = self.com()
        distances = [np.linalg.norm(particle.pos() - COM) + particle.R for particle in self]
        return max(distances)

    # Calculate volume occupied by assembly particles. Uses convex hull.
    # Clearly, usefullness and accuracy will depend on the inputs.
    def vol(self):
        if self.N() < 4:
            raise ValueError("Volume calculation requires at least 4 particles. See scipy.spatial.ConvexHull() for docs.")
        points = [(particle.x, particle.y, particle.z) for particle in self]
        hull = ss.ConvexHull(points)
        return hull.volume


    # Average particle density, as opposed to bulk density below. Confusing, consider removing.
    def avg_dens(self):
        dens = []
        for particle in self:
            dens.append(particle.m/((4/3)*np.pi*particle.R**3))
        return np.average(dens)

    def bulk_dens(self):
        if self.N() < 4:
            raise TypeError("Volume calculation requires at least 4 particles. See scipy.spatial.ConvexHull() for docs.")
        return self.M()/self.vol()

    # Return the inertia tensor of the assembly. Taken from pkdgrav function pkdAggsGetAxesAndSpin() in aggs.c.
    def I(self):
        com = self.com()
        comv = self.comv()

        I_matrix = np.zeros((3,3))

        for particle in self:
            # Save particle's mass, radius, moment of inertia prefactor, relative position, and relative velocity for ease of access.
            m = particle.m
            R = particle.R
            q = 0.4*R**2
            r = particle.pos() - com
            v = particle.vel() - comv

            I_matrix[0][0] += m*(q + r[1]**2 + r[2]**2)
            I_matrix[0][1] -= m*r[0]*r[1]
            I_matrix[0][2] -= m*r[0]*r[2]
            I_matrix[1][1] += m*(q + r[0]**2 + r[2]**2)
            I_matrix[1][2] -= m*r[1]*r[2]
            I_matrix[2][2] += m*(q + r[0]**2 + r[1]**2)

        I_matrix[1][0] = I_matrix[0][1]
        I_matrix[2][0] = I_matrix[0][2]
        I_matrix[2][1] = I_matrix[1][2]

        return I_matrix

    # Return the principal axes of the assembly. Just get the eigenvectors of the inertia tensor.
    def axes(self):
        I = self.I()
        eigenvals, eigenvecs = np.linalg.eig(I)

        idx = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:,idx]

        return eigenvecs

    # Return the angular momentum vector of the assembly.
    def L(self):
        com = self.com()
        comv = self.comv()

        L_vector = np.zeros(3)

        for particle in self:
            m = particle.m
            R = particle.R
            q = 0.4*R**2
            r = particle.pos() - com
            v = particle.vel() - comv
            w = particle.spin()
            p = m*v

            L_vector += np.cross(r, p)
            L_vector += q*m*w

        #L_vector.sort()
        return L_vector

    # Calculate the DEEVE semi-axes, equivalent radius, and bulk density. Formulae taken from rpu.c and rpx.c.
    def deeve(self):
        M = self.M()
        I = self.I()
        moments = [I[0][0], I[1][1], I[2][2]]
        moments.sort()

        semi_axes = [np.sqrt(2.5 * (moments[1] + moments[2] - moments[0]) / M),
                     np.sqrt(2.5 * (moments[0] + moments[2] - moments[1]) / M),
                     np.sqrt(2.5 * (moments[0] + moments[1] - moments[2]) / M)]

        semi_axes.sort()

        radius =  (semi_axes[0] * semi_axes[1] * semi_axes[2])**(1/3)
        bulk_dens = M / ((4/3) * np.pi * radius**3)

        return np.array(semi_axes), radius, bulk_dens

    # Calculate the lengths of the three semi-axes of an assembly. Of course, this operation will only make sense for a 
    # roughly ellipsoidal rubble pile. This was added to support the new rubble pile analysis module---more testing is
    # warranted.
    def semi_axes(self):
        # Singlet assembly's create weird issues, but must be accepted to work with find_rp().
        if self.N() == 1:
            R = self[0].R
            return np.array([R, R, R])

        semi = np.zeros(3)
        vMin = np.zeros(3)
        vMax = np.zeros(3)

        com = self.com()
        axes = self.axes()
        # Needed to give correct results below. This needs a more careful look. Maybe the axes() method should return this
        # transposed matrix directly. No time to follow up on this at the moment...
        axes = np.transpose(axes)

        for particle in self:
            r = particle.pos() - com
            s = np.dot(axes, r)

            for i in range(3):
                if s[i] - particle.R < vMin[i]:
                    vMin[i] = s[i] - particle.R
                if s[i] + particle.R > vMax[i]:
                    vMax[i] = s[i] + particle.R

        for i in range(3):
            semi[i] = 0.5*(vMax[i] - vMin[i])

        semi.sort()
        return semi

    # Probably superfluous after adding __str__
    def show_particles(self):
        for particle in self:
            print(particle)

    # Add a copy of a particle to an assembly. Future manipulations of the assembly will not affect the original particle.
    # Not sure this is the right approach, this behavior may change in the future.
    def add_particles(self, *particles):
        for element in particles:
            if not isinstance(element, Particle):
                raise TypeError("Can only add particles to assemblies.")
            self.append(element.copy())

    # This will return a *new* particle with the same properties as the one requested. To pick out the specified particle itself
    # from the assembly to edit, user can use list slicing. E.g.: <assembly>.get_particle(7) will return a copy of the particle
    # with iOrder 7 in the agg.
    def get_particle(self, iOrder):
        for particle in self:
            if particle.iOrder == iOrder:
                return particle.copy()
        print("No particle with the given iOrder was found.")

    # When calling, need to unpack any list arguments: <assembly>.del_particles(*<list with particles to be deleted>)
    def del_particles(self, *iOrders):
        for element in iOrders:
            if not isinstance(element, int):
                raise TypeError("del_particles() can only take integers (iOrder values) as arguments.\n"
                                "If you would like to use a list to specify the particles to be deleted, use the '*' operator.\n" 
                                "E.g.: <assembly>.del_particles(*<list with particles to be deleted>)")
        # Need to do it this way. Deleting elements while iterating over a list is not good. Create new list w/ desired 
        # particles and overwrite existing assembly.
        self[:] = Assembly(*[particle for particle in self if particle.iOrder not in iOrders], units=self.units, time=self.time)

    def copy(self):
        return Assembly(*[particle.copy() for particle in self], units=self.units, time=self.time)

    # Sort assmblies by iOrder. Optional 'direction' argument allows sorting in ascending or descending order.
    def sort_iOrder(self, direction='a'):
        if direction == 'a':
            self.sort(key=util.iOrder_key)
        elif direction == 'd':
            self.sort(key=util.iOrder_key, reverse=True)
        else:
            raise ValueError("Error: direction argument can be 'a' for ascending or 'd' for descending.  Default is ascending.") 

    # Similar to the preceding function, but with opposite default behavior.
    def sort_iOrgIdx(self, direction='d'):
        if direction == 'a':
            self.sort(key=util.iOrgIdx_key)
        elif direction == 'd':
            self.sort(key=util.iOrgIdx_key, reverse=True)
        else:
            raise ValueError("Error: direction argument can be 'a' for ascending or 'd' for descending. Default is descending.")

    # Renumber iOrders consecutively, either ascending or descending.
    def condense_iOrder(self, direction='a'):
        self.sort_iOrder()
        for i, particle in enumerate(self):
            particle.iOrder = i

        if direction == 'd':
            self.sort_iOrder('d')

    def condense(self, direction='a'):
        self.condense_aggs()
        self.condense_iOrder(direction)

    # Rotate entire assembly by the specified angle about the specified axis. Note that this will be a rotation
    # about the origin regardless of where the assembly is centered. Maybe add something to allow user to specify in the future.
    def rotate(self, axis, angle):
        # Kludge to prevent weird stuff from happening in these cases. If angle is
        # zero or no legit axis is specified, do nothing.
        if np.linalg.norm(axis) == 0 or angle == 0:
            raise ValueError("Warning: invalid axis or angle of rotation passed to rotate() method. No rotation was performed.")

        for particle in self:
            rotated = util.vector_rotate(particle.pos(), axis, angle)

            particle.x = rotated[0]
            particle.y = rotated[1]
            particle.z = rotated[2]

    # Allows sensible printing of assembly contents.
    def __str__(self):
        to_print = ''
        for particle in self:
            to_print += (f'[{particle.iOrder}, {particle.iOrgIdx}, {particle.m:E}, {particle.R:E}, {particle.x:E}, {particle.y:E}, {particle.z:E}, '
                         f'{particle.vx:E}, {particle.vy:E}, {particle.vz:E}, {particle.wx:E}, {particle.wy:E}, {particle.wz:E}, {particle.color}, {particle.units}]\n')
        # Add time value to print string.
        to_print += (f'{self.time}')
        # Remove final new line character before returning.
        return to_print
