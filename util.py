##################################################################
# util.py
# Julian C. Marohnic
# Created: 8/16/23
# 
# Utility functions that don't logically belong to one of the 
# primary modules. Adapted from earlier work (c. 2022-23) 
# by JCM and JVD.
##################################################################

import numpy as np

# Key function for sorting assemblies by iOrder
def iOrder_key(particle):
    return particle.iOrder

# Key function for sorting assemblies by iOrgIdx
def iOrgIdx_key(particle):
    return particle.iOrgIdx

### UNIT FUNCTIONS ###

# Base conversions. l, m, and t are length, mass, and time, respectively.
l_pkd2cgs = 1.495978707e13
l_pkd2mks = 1.495978707e11
l_mks2cgs = 1.0e2
m_pkd2cgs = 1.98847e33
m_pkd2mks = 1.98847e30
m_mks2cgs = 1.0e3
t_pkd2cgs = 5.02254803e6
t_pkd2mks = 5.02254803e6

# Tedious unit conversion utilities. Could make this a little slicker, but I've opted for
# simplicity and ease of debugging. Numbers above courtesy of Wikipedia.
def pkd2cgs(particle):
    particle.m = m_pkd2cgs*particle.m
    particle.R = l_pkd2cgs*particle.R
    particle.x = l_pkd2cgs*particle.x
    particle.y = l_pkd2cgs*particle.y
    particle.z = l_pkd2cgs*particle.z
    particle.vx = (l_pkd2cgs/t_pkd2cgs)*particle.vx
    particle.vy = (l_pkd2cgs/t_pkd2cgs)*particle.vy
    particle.vz = (l_pkd2cgs/t_pkd2cgs)*particle.vz
    particle.wx = (1.0/t_pkd2cgs)*particle.wx
    particle.wy = (1.0/t_pkd2cgs)*particle.wy
    particle.wz = (1.0/t_pkd2cgs)*particle.wz
    particle._units = 'cgs'

def cgs2pkd(particle):
    particle.m = (1.0/m_pkd2cgs)*particle.m
    particle.R = (1.0/l_pkd2cgs)*particle.R
    particle.x = (1.0/l_pkd2cgs)*particle.x
    particle.y = (1.0/l_pkd2cgs)*particle.y
    particle.z = (1.0/l_pkd2cgs)*particle.z
    particle.vx = (t_pkd2cgs/l_pkd2cgs)*particle.vx
    particle.vy = (t_pkd2cgs/l_pkd2cgs)*particle.vy
    particle.vz = (t_pkd2cgs/l_pkd2cgs)*particle.vz
    particle.wx = t_pkd2cgs*particle.wx
    particle.wy = t_pkd2cgs*particle.wy
    particle.wz = t_pkd2cgs*particle.wz
    particle._units = 'pkd'

def pkd2mks(particle):
    particle.m = m_pkd2mks*particle.m
    particle.R = l_pkd2mks*particle.R
    particle.x = l_pkd2mks*particle.x
    particle.y = l_pkd2mks*particle.y
    particle.z = l_pkd2mks*particle.z
    particle.vx = (l_pkd2mks/t_pkd2mks)*particle.vx
    particle.vy = (l_pkd2mks/t_pkd2mks)*particle.vy
    particle.vz = (l_pkd2mks/t_pkd2mks)*particle.vz
    particle.wx = (1.0/t_pkd2mks)*particle.wx
    particle.wy = (1.0/t_pkd2mks)*particle.wy
    particle.wz = (1.0/t_pkd2mks)*particle.wz
    particle._units = 'mks'

def mks2pkd(particle):
    particle.m = (1.0/m_pkd2mks)*particle.m
    particle.R = (1.0/l_pkd2mks)*particle.R
    particle.x = (1.0/l_pkd2mks)*particle.x
    particle.y = (1.0/l_pkd2mks)*particle.y
    particle.z = (1.0/l_pkd2mks)*particle.z
    particle.vx = (t_pkd2mks/l_pkd2mks)*particle.vx
    particle.vy = (t_pkd2mks/l_pkd2mks)*particle.vy
    particle.vz = (t_pkd2mks/l_pkd2mks)*particle.vz
    particle.wx = t_pkd2mks*particle.wx
    particle.wy = t_pkd2mks*particle.wy
    particle.wz = t_pkd2mks*particle.wz
    particle._units = 'pkd'

def mks2cgs(particle):
    particle.m = m_mks2cgs*particle.m
    particle.R = l_mks2cgs*particle.R
    particle.x = l_mks2cgs*particle.x
    particle.y = l_mks2cgs*particle.y
    particle.z = l_mks2cgs*particle.z
    particle.vx = l_mks2cgs*particle.vx
    particle.vy = l_mks2cgs*particle.vy
    particle.vz = l_mks2cgs*particle.vz
    particle._units = 'cgs'

def cgs2mks(particle):
    particle.m = (1.0/m_mks2cgs)*particle.m
    particle.R = (1.0/l_mks2cgs)*particle.R
    particle.x = (1.0/l_mks2cgs)*particle.x
    particle.y = (1.0/l_mks2cgs)*particle.y
    particle.z = (1.0/l_mks2cgs)*particle.z
    particle.vx = (1.0/l_mks2cgs)*particle.vx
    particle.vy = (1.0/l_mks2cgs)*particle.vy
    particle.vz = (1.0/l_mks2cgs)*particle.vz
    particle._units = 'mks'

def pkd2sec(assembly):
    assembly.time = t_pkd2mks*assembly.time

def sec2pkd(assembly):
    assembly.time = (1.0/t_pkd2mks)*assembly.time

# General rotation of a vector, expressed in terms of axis and angle. Copied from ssgen2Agg.py/Wikipedia.
# We'll use this for specifying the orientation when we generate aggs. Investigate the merits of a
# quaternion approach? Note that this will be a rotation about the origin regardless of where the assembly
# is centered. Maybe something to allow user to specify in the future.
def vector_rotate(vector, axis, angle):
    sa = np.sin(angle)
    ca = np.cos(angle)
    dot = np.dot(vector, axis)
    x = vector[0]
    y = vector[1]
    z = vector[2]
    u = axis[0]
    v = axis[1]
    w = axis[2]

    rotated = np.zeros(3)

    rotated[0] = u*dot+(x*(v*v+w*w) - u*(v*y+w*z))*ca + (-w*y+v*z)*sa
    rotated[1] = v*dot+(y*(u*u+w*w) - v*(u*x+w*z))*ca + (w*x-u*z)*sa
    rotated[2] = w*dot+(z*(u*u+v*v) - w*(u*x+v*y))*ca + (-v*x+u*y)*sa

    return rotated

# Calculate the angle between two vectors. Used when setting orientation of generated aggs. Copied from a post on Stack Overflow:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Return the coordinates for plotting a sphere centered at (x,y,z)
# Taken from Stack Overflow post: https://stackoverflow.com/questions/70977042/how-to-plot-spheres-in-3d-with-plotly-or-another-library
def makesphere(x, y, z, radius, resolution=10):
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

# Single-use utility for converting pkd colors to matplotlib colors in viz().
def color_translate(pkd_color):
    if pkd_color == 0:
        return 'black'
    elif pkd_color == 1:
        return 'white'
    elif pkd_color == 2:
        return 'red'
    elif pkd_color == 3:
        return 'lawngreen'
    elif pkd_color == 4:
        return 'blue'
    elif pkd_color == 5:
        return 'yellow'
    elif pkd_color == 6:
        return 'magenta'
    elif pkd_color == 7:
        return 'cyan'
    elif pkd_color == 8:
        return 'gold'
    elif pkd_color == 9:
        return 'pink'
    elif pkd_color == 10:
        return 'orange'
    elif pkd_color == 11:
        return 'khaki'
    elif pkd_color == 12:
        return 'mediumpurple'
    elif pkd_color == 13:
        return 'maroon'
    elif pkd_color == 14:
        return 'aqua'
    elif pkd_color == 15:
        return 'navy'
    elif pkd_color == 16:
        return 'black'
    else:
        return 'gray'
