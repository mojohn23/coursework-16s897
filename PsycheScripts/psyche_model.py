import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

density = 113.62 # [kg/m^3]

class BoxObj:
    def __init__(self, height: float, width: float, depth: float):
        # (depth, width, height) = (x, y, z)
        self.volume = height*width*depth # [m^3]
        self.mass = density*self.volume # [kg]
        self.com = np.transpose(np.array([depth/2, width/2, height/2], ndmin = 2)) # [m]
        self.geo = self.com
        Ixx = 1/12*(width**2 + height**2)
        Iyy = 1/12*(depth**2 + height**2)
        Izz = 1/12*(depth**2 + width**2)
        self.MOI = self.mass*np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.MOICOM = None # placeholder
        self.sax = width*height # surface area on +/- X face
        self.say = depth*height # surface area on +/- Y face
        self.saz = width*depth # surface area on +/- Z face
        self.nvecx1 = np.transpose(np.array([1, 0, 0], ndmin = 2)) # normal vector on + X face
        self.nvecx2 = np.transpose(np.array([-1, 0, 0], ndmin = 2)) # normal vector on - X face
        self.nvecy1 = np.transpose(np.array([0, 1, 0], ndmin = 2)) # normal vector on + Y face
        self.nvecy2 = np.transpose(np.array([0, -1, 0], ndmin = 2)) # normal vector on - Y face
        self.nvecz1 = np.transpose(np.array([0, 0, 1], ndmin = 2)) # normal vector on + Z face
        self.nvecz2 = np.transpose(np.array([0, 0, -1], ndmin = 2)) # normal vector on + Z face

class FrustumObj:
    def __init__(self, r1: float, r2: float, height: float):
        # note that r1 > r2
        if r1 < r2:
            raise ValueError('Error: r1 > r2, r1 is the base of the frustum')
        self.volume = 1/3*m.pi*height*(r1**2 + r2**2 + (r1*r2))
        self.mass = density*self.volume # [kg]
        self.com = np.transpose(np.array([0, 0, (height*((r1 + r2)**2 + 2*r2**2))/(4*((r1 + r2)**2 - r1*r2))], ndmin = 2)) # [m]
        self.geo = self.com
        Ixx = 1 # Placeholder
        Iyy = Ixx
        Izz = 3/10*(r1**5 - r2**5)/(r1**3 - r2**3)
        self.MOI = self.mass*np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.MOICOM = None # [m^2], placeholder
        # Surface area and normal vectors assume a slanted rectangular prism!
        self.sax = height/2*(r1 + r2) # surface area on +/- X face
        self.say = height/2*(r1 + r2) # surface area on +/- Y face
        self.saz1 = r2**2 # surface area on +Z face
        self.saz2 = r1**2 # surface area on -Z face
        
        slant = m.sqrt((r1 - r2)**2 + height**2) # [m]
        slant_angle = m.acos((r1 - r2)/slant) # [rad]
        Y = rot_simple(-slant_angle, 'y', 'rad')
        self.nvecx1 = Y@np.transpose(np.array([1, 0, 0], ndmin = 2)) # normal vector on + X face
        Y = rot_simple(slant_angle, 'y', 'rad')
        self.nvecx2 = Y@np.transpose(np.array([-1, 0, 0], ndmin = 2)) # normal vector on - X face
        X = rot_simple(slant_angle, 'x', 'rad')
        self.nvecy1 = X@np.transpose(np.array([0, 1, 0], ndmin = 2)) # normal vector on + Y face
        X = rot_simple(-slant_angle, 'x', 'rad')
        self.nvecy2 = X@np.transpose(np.array([0, -1, 0], ndmin = 2)) # normal vector on + Y face
        self.nvecz1 = np.transpose(np.array([0, 0, 1], ndmin = 2)) # normal vector on + Z face
        self.nvecz2 = np.transpose(np.array([0, 0, -1], ndmin = 2)) # normal vector on + Z face

class RodObj:
    def __init__(self, radius:float, length:float):
        self.volume = m.pi*radius**2*length # [m^3]
        self.mass = density*self.volume # [kg]
        self.com = np.transpose(np.array([0, length/2, 0], ndmin = 2)) # [m]
        self.geo = self.com
        Ixx = 1/12*length**2 + 1/4*radius**2
        Iyy = 1/2*radius**2
        Izz = 1/12*length**2 + 1/4*radius**2
        self.MOI = self.mass*np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.MOICOM = None # placeholder
        # Surface area and normal vectors assume a rectangular prism!
        self.sax = length*2*radius # surface area on +/- X face
        self.say = (2*radius)**2 # surface area on +/- Y face
        self.saz = length*2*radius # surface area on +/- Z face
        self.nvecx1 = np.transpose(np.array([1, 0, 0], ndmin = 2)) # normal vector on + X face
        self.nvecx2 = np.transpose(np.array([-1, 0, 0], ndmin = 2)) # normal vector on - X face
        self.nvecy1 = np.transpose(np.array([0, 1, 0], ndmin = 2)) # normal vector on + Y face
        self.nvecy2 = np.transpose(np.array([0, -1, 0], ndmin = 2)) # normal vector on - Y face
        self.nvecz1 = np.transpose(np.array([0, 0, 1], ndmin = 2)) # normal vector on + Z face
        self.nvecz2 = np.transpose(np.array([0, 0, -1], ndmin = 2)) # normal vector on + Z face

def rot_simple(angle: float, axis: str, unit:str = 'deg'):
    axis = axis.upper()
    unit = unit.upper()

    # Need to convert from degrees to rad for math trig functions
    if unit in ('DEG', 'DEGREE', 'DEGREES'):
        theta = m.radians(angle)
    elif unit in ('RAD', 'RADIAN', 'RADIANS'):
        theta = angle
    else:
        raise ValueError('Error: Unit must be deg or rad, deg by default')

    c = m.cos(theta)
    s = m.sin(theta)

    if axis == 'X':
        rot_mat = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'Y':
        rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'Z':
        rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError('Error: Axis must be X, Y, or Z')
    return rot_mat

def parallel_axis(MOI, mass: float, distance):
    # moi is the moment of inertia about the LOCAL center of mass
    # output MOICOM is the moment of inertia about the GLOBAL center of mass
    # distance is local COM minus global COM, input as a 3x1 vector
    MOICOM = MOI + mass*((distance.T@distance)*np.eye(3) - (distance@distance.T))
    return MOICOM

# I am hiding long sections of code in these if statements because it lets me collapse the lines lmao

# Initialize each piece with volume, mass, body MOI, (approximated) XYZ surface areas, (approximated) surface normal vectors
if True:
    bus_height = 2.9 # [m] Z
    bus_width = 2.2 # [m] Y
    bus_depth = 2.4 # [m] X
    
    ##### MAIN BODY ######################################################################################
    # Bus
    bus = BoxObj(bus_height, bus_width, bus_depth)        
    # Antenna bottom, rotated 180 and placed on body +Z face
    antenna1 = FrustumObj(1, 0.5, 0.5)
    # Antenna top, placed on top of antenna bottom
    antenna2 = FrustumObj(1, 0.1, 0.25)

    ##### SUPPORTS +Y ####################################################################################
    # Rod1 rotated +45 and placed along body +Y face
    rod1L = RodObj(0.025, 1.6829)
    # Rod2 rotated -45 and placed along body +Y face
    rod2L = RodObj(0.025, 1.6829)
    # Rod3 rotated 90 and placed at the tip of rod1 and rod2
    rod3L = RodObj(0.025, 2.4)
    # Rod4 placed at the tip of rod1
    rod4L = RodObj(0.025, 0.55)
    # Rod5 placed at the tip of rod3
    rod5L = RodObj(0.025, 0.55)

    ##### SOLAR PANELS +Y ##############################################################################
    panelmidL = BoxObj(0.1, 3.18, 2.4)
    panelouterL = BoxObj(0.1, 3.18, 2.4)
    panelinnerL = BoxObj(0.1, 3.18, 2.4)
    panelupL = BoxObj(0.1, 3.18, 2.4)
    paneldownL = BoxObj(0.1, 3.18, 2.4)

    ##### SUPPORTS -Y #################################################################################
    # Rod1 rotated -45 and placed along body -Y face
    rod1R = RodObj(0.025, 1.6829)
    # Rod2 rotated 45 and placed along body -Y face
    rod2R = RodObj(0.025, 1.6829)
    # Rod3 rotated 90 and placed at the tip of rod1 and rod2
    rod3R = RodObj(0.025, 2.4)
    # Rod4 placed at the tip of rod1
    rod4R = RodObj(0.025, 0.55)
    # Rod5 placed at the tip of rod3
    rod5R = RodObj(0.025, 0.55)

    ##### SOLAR PANELS -Y #############################################################################
    panelmidR = BoxObj(0.1, 3.18, 2.4)
    panelouterR = BoxObj(0.1, 3.18, 2.4)
    panelinnerR = BoxObj(0.1, 3.18, 2.4)
    panelupR = BoxObj(0.1, 3.18, 2.4)
    paneldownR = BoxObj(0.1, 3.18, 2.4)

# Calculate the COM of each piece and then for the whole
if True:
    ##### MAIN BODY ######################################################################################
    bus.com = np.transpose(np.array([0, 0, 0], ndmin = 2)) # Assume bus COM is the origin
    antenna1.com = np.transpose(np.array([0, 0, bus_height/2 + 0.5], ndmin = 2)) - antenna1.com
    antenna2.com = antenna2.com + np.transpose(np.array([0, 0, bus_height/2 + 0.5], ndmin = 2))

    ##### SUPPORTS +Y ####################################################################################
    # Rod1 rotated +45 and placed along body +Y face
    Z = rot_simple(45, 'z')
    rod1L.com = Z@rod1L.com + np.transpose(np.array([0, bus_width/2, 0], ndmin = 2))
    # Rod2 rotated -45 and placed along body +Y face
    Z = rot_simple(-45, 'z')
    rod2L.com = Z@rod2L.com + np.transpose(np.array([0, bus_width/2, 0], ndmin = 2))
    # Rod3 rotated 90 and placed at the tip of rod1 and rod2
    rod3L.com = np.transpose(np.array([0, bus_width/2 + 1.2, 0], ndmin = 2))
    # Rod4 placed at the tip of rod1
    rod4L.com = rod4L.com + np.transpose(np.array([-bus_depth/2, bus_width/2 + 1.2, 0], ndmin = 2))
    # Rod5 placed at the tip of rod3
    rod5L.com = rod5L.com + np.transpose(np.array([bus_depth/2, bus_width/2 + 1.2, 0], ndmin = 2))

    ##### SOLAR PANELS +Y ##############################################################################
    panelmidL.com = np.transpose(np.array([0, bus_width/2 + 1.74 + 3.18 + 3.18/2, 0], ndmin = 2))
    panelouterL.com = np.transpose(np.array([0, bus_width/2 + 1.74 + 2*3.18 + 3.18/2, 0], ndmin = 2))
    panelinnerL.com = np.transpose(np.array([0, bus_width/2 + 1.74 + 3.18/2, 0], ndmin = 2))
    panelupL.com = np.transpose(np.array([-(bus_depth/2 + 2.4/2), bus_width/2 + 1.74 + 3.18 + 3.18/2, 0], ndmin = 2))
    paneldownL.com = np.transpose(np.array([(bus_depth/2 + 2.4/2), bus_width/2 + 1.74 + 3.18 + 3.18/2, 0], ndmin = 2))
    
    ##### SUPPORTS -Y #################################################################################
    # Rod1 rotated -45 and placed along body -Y face
    Z = rot_simple(-45, 'z')
    rod1R.com = Z@rod1R.com + np.transpose(np.array([0, -bus_width/2, 0], ndmin = 2))
    # Rod2 rotated 45 and placed along body -Y face
    Z = rot_simple(45, 'z')
    rod2R.com = Z@rod2R.com + np.transpose(np.array([0, -bus_width/2, 0], ndmin = 2))
    # Rod3 rotated 90 and placed at the tip of rod1 and rod2
    rod3R.com = np.transpose(np.array([0, -bus_width/2 - 1.2, 0], ndmin = 2))
    # Rod4 placed at the tip of rod1
    rod4R.com = rod4R.com + np.transpose(np.array([-bus_depth/2, -bus_width/2 - 1.2, 0], ndmin = 2))
    # Rod5 placed at the tip of rod3
    rod5R.com = rod5R.com + np.transpose(np.array([bus_depth/2, -bus_width/2 - 1.2, 0], ndmin = 2))

    ##### SOLAR PANELS -Y #############################################################################
    panelmidR.com = np.transpose(np.array([0, -(bus_width/2 + 1.74 + 3.18 + 3.18/2), 0], ndmin = 2))
    panelouterR.com = np.transpose(np.array([0, -(bus_width/2 + 1.74 + 2*3.18 + 3.18/2), 0], ndmin = 2))
    panelinnerR.com = np.transpose(np.array([0, -(bus_width/2 + 1.74 + 3.18/2), 0], ndmin = 2))
    panelupR.com = np.transpose(np.array([-(bus_depth/2 + 2.4/2), -(bus_width/2 + 1.74 + 3.18 + 3.18/2), 0], ndmin = 2))
    paneldownR.com = np.transpose(np.array([(bus_depth/2 + 2.4/2), -(bus_width/2 + 1.74 + 3.18 + 3.18/2), 0], ndmin = 2))

    ##### Full spacecraft #############################################################################
    tot_bus_vol = bus.volume + antenna1.volume + antenna2.volume
    tot_left_vol = rod1L.volume + rod2L.volume + rod3L.volume + rod4L.volume + rod5L.volume + panelmidL.volume + panelouterL.volume + panelinnerL.volume + panelupL.volume + paneldownL.volume
    tot_right_vol = rod1R.volume + rod2R.volume + rod3R.volume + rod4R.volume + rod5R.volume + panelmidR.volume + panelouterR.volume + panelinnerR.volume + panelupR.volume + paneldownR.volume
    tot_vol = tot_bus_vol + tot_left_vol + tot_right_vol
    
    tot_mass = density*tot_vol
    
    tot_bus_com = bus.mass*bus.com + antenna1.mass*antenna1.com + antenna2.mass*antenna2.com
    tot_left_com = rod1L.mass*rod1L.com + rod2L.mass*rod2L.com + rod3L.mass*rod3L.com + rod4L.mass*rod4L.com + rod5L.mass*rod5L.com + panelmidL.mass*panelmidL.com + panelouterL.mass*panelouterL.com + panelinnerL.mass*panelinnerL.com + panelupL.mass*panelupL.com + paneldownL.mass*paneldownL.com
    tot_right_com = rod1R.mass*rod1R.com + rod2R.mass*rod2R.com + rod3R.mass*rod3R.com + rod4R.mass*rod4R.com + rod5R.mass*rod5R.com + panelmidR.mass*panelmidR.com + panelouterR.mass*panelouterR.com + panelinnerR.mass*panelinnerR.com + panelupR.mass*panelupR.com + paneldownR.mass*paneldownR.com
    tot_com = (tot_bus_com + tot_left_com + tot_right_com)/tot_mass
    tot_com[1] = 0 # Value is off by 0.0003744 due to rounding errors, force this to be zero for consistency
    
# ignore this section this was just a visual sanity check for the COM calculations
if False:
    print('ignore')
    ## Plot the individual COM to see if it looks semi-legit I'M TWEAKING HOW DOES ANYONE USE PYTHON
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')

    # ax.plot(bus.com[0], bus.com[1], bus.com[2], 'o')
    # ax.plot(antenna1.com[0], antenna1.com[1], antenna1.com[2], 'o')
    # ax.plot(antenna2.com[0], antenna2.com[1], antenna2.com[2], 'o')

    # ax.plot(rod1L.com[0], rod1L.com[1], rod1L.com[2], 'ro')
    # ax.plot(rod2L.com[0], rod2L.com[1], rod2L.com[2], 'ro')
    # ax.plot(rod3L.com[0], rod3L.com[1], rod3L.com[2], 'ro')
    # ax.plot(rod4L.com[0], rod4L.com[1], rod4L.com[2], 'ro')
    # ax.plot(rod5L.com[0], rod5L.com[1], rod5L.com[2], 'ro')

    # ax.plot(panelmidL.com[0], panelmidL.com[1], panelmidL.com[2], 'go')
    # ax.plot(panelouterL.com[0], panelouterL.com[1], panelouterL.com[2], 'go')
    # ax.plot(panelinnerL.com[0], panelinnerL.com[1], panelinnerL.com[2], 'go')
    # ax.plot(panelupL.com[0], panelupL.com[1], panelupL.com[2], 'go')
    # ax.plot(paneldownL.com[0], paneldownL.com[1], paneldownL.com[2], 'go')

    # ax.plot(rod1R.com[0], rod1R.com[1], rod1R.com[2], 'ro')
    # ax.plot(rod2R.com[0], rod2R.com[1], rod2R.com[2], 'ro')
    # ax.plot(rod3R.com[0], rod3R.com[1], rod3R.com[2], 'ro')
    # ax.plot(rod4R.com[0], rod4R.com[1], rod4R.com[2], 'ro')
    # ax.plot(rod5R.com[0], rod5R.com[1], rod5R.com[2], 'ro')

    # ax.plot(panelmidR.com[0], panelmidR.com[1], panelmidR.com[2], 'go')
    # ax.plot(panelouterR.com[0], panelouterR.com[1], panelouterR.com[2], 'go')
    # ax.plot(panelinnerR.com[0], panelinnerR.com[1], panelinnerR.com[2], 'go')
    # ax.plot(panelupR.com[0], panelupR.com[1], panelupR.com[2], 'go')
    # ax.plot(paneldownR.com[0], paneldownR.com[1], paneldownR.com[2], 'go')

    # ax.set_ylim(-10, 10)
    # ax.set_xlim(-10, 10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

# Calculate the MOI at the global COM for each piece and then for the whole
if True:
    ##### MAIN BODY ######################################################################################
    bus.MOICOM = parallel_axis(bus.MOI, bus.mass, bus.com - tot_com)
    # antenna1 is rotated 180 about X (upside-down)
    X = rot_simple(180, 'x')
    antenna1.MOI = X@antenna1.MOI@X.T
    antenna1.MOICOM = parallel_axis(antenna1.MOI, antenna1.mass, antenna1.com - tot_com)
    antenna2.MOICOM = parallel_axis(antenna2.MOI, antenna2.mass, antenna2.com - tot_com)
    
    ##### +Y SIDE #######################################################################################
    # rod1 is rotated +45, rod2 is -45, rod3 is +90 (all about Z)
    Z = rot_simple(45, 'z')
    rod1L.MOI = Z@rod1L.MOI@Z.T # The MOI about the local COM must be rotated to the body frame since the rod itself is rotated
    rod1L.MOICOM = parallel_axis(rod1L.MOI, rod1L.mass, rod1L.com - tot_com)
    Z = rot_simple(-45, 'z')
    rod2L.MOI = Z@rod2L.MOI@Z.T # The MOI about the local COM must be rotated to the body frame since the rod itself is rotated
    rod2L.MOICOM = parallel_axis(rod2L.MOI, rod2L.mass, rod2L.com - tot_com)
    
    Z = rot_simple(90, 'z')
    rod3L.MOI = Z@rod3L.MOI@Z.T # The MOI about the local COM must be rotated since the rod itself is rotated
    rod3L.MOICOM = parallel_axis(rod3L.MOI, rod3L.mass, rod3L.com - tot_com)

    rod4L.MOICOM = parallel_axis(rod4L.MOI, rod4L.mass, rod4L.com - tot_com)
    rod5L.MOICOM = parallel_axis(rod5L.MOI, rod5L.mass, rod5L.com - tot_com)
    panelmidL.MOICOM = parallel_axis(panelmidL.MOI, panelmidL.mass, panelmidL.com - tot_com)
    panelouterL.MOICOM = parallel_axis(panelouterL.MOI, panelouterL.mass, panelouterL.com - tot_com)
    panelinnerL.MOICOM = parallel_axis(panelinnerL.MOI, panelinnerL.mass, panelinnerL.com - tot_com)
    panelupL.MOICOM = parallel_axis(panelupL.MOI, panelupL.mass, panelupL.com - tot_com)
    paneldownL.MOICOM = parallel_axis(paneldownL.MOI, paneldownL.mass, paneldownL.com - tot_com)

    ##### -Y SIDE ########################################################################################
    # rod1 is rotated -45, rod2 is +45, rod 3 is +90
    Z = rot_simple(-45, 'z')
    rod1R.MOI = Z@rod1R.MOI@Z.T # The MOI about the local COM must be rotated to the body frame since the rod itself is rotated
    rod1R.MOICOM = parallel_axis(rod1R.MOI, rod1R.mass, rod1R.com - tot_com)
    Z = rot_simple(45, 'z')
    rod2R.MOI = Z@rod2R.MOI@Z.T # The MOI about the local COM must be rotated to the body frame since the rod itself is rotated
    rod2R.MOICOM = parallel_axis(rod2R.MOI, rod2R.mass, rod2R.com - tot_com)
    
    Z = rot_simple(90, 'z')
    rod3R.MOI = Z@rod3R.MOI@Z.T # The MOI about the local COM must be rotated since the rod itself is rotated
    rod3R.MOICOM = parallel_axis(rod3R.MOI, rod3R.mass, rod3R.com - tot_com)

    rod4R.MOICOM = parallel_axis(rod4R.MOI, rod4R.mass, rod4R.com - tot_com)
    rod5R.MOICOM = parallel_axis(rod5R.MOI, rod5R.mass, rod5R.com - tot_com)
    panelmidR.MOICOM = parallel_axis(panelmidR.MOI, panelmidR.mass, panelmidR.com - tot_com)
    panelouterR.MOICOM = parallel_axis(panelouterR.MOI, panelouterR.mass, panelouterR.com - tot_com)
    panelinnerR.MOICOM = parallel_axis(panelinnerR.MOI, panelinnerR.mass, panelinnerR.com - tot_com)
    panelupR.MOICOM = parallel_axis(panelupR.MOI, panelupR.mass, panelupR.com - tot_com)
    paneldownR.MOICOM = parallel_axis(paneldownR.MOI, paneldownR.mass, paneldownR.com - tot_com)

    ##### TOTAL SPACECRAFT #################################################################################
    tot_bus_moicom = bus.MOICOM + antenna1.MOICOM + antenna2.MOICOM
    tot_left_moicom = rod4L.MOICOM + rod5L.MOICOM + panelmidL.MOICOM + panelouterL.MOICOM + panelinnerL.MOICOM + panelupL.MOICOM + paneldownL.MOICOM
    tot_right_moicom = rod4R.MOICOM + rod5R.MOICOM + panelmidR.MOICOM + panelouterR.MOICOM + panelinnerR.MOICOM + panelupR.MOICOM + paneldownR.MOICOM
    tot_moicom = tot_bus_moicom + tot_left_moicom + tot_right_moicom
    tot_moicom[1, 2] = 0 # very close to 0, should be 0 because of symmetry
    tot_moicom[2, 1] = 0 # very close to 0, should be 0 because of symmetry

# Align the surface normal vectors for rotated pieces, and remove "blocked" faces
if True:
##### MAIN BODY ######################################################################################
    bus.nvecz1 = None # Blocked by antenna
    # antenna1 is rotated 180 about X (upside-down)
    X = rot_simple(180, 'x')
    antenna1.nvecx1 = X@antenna1.nvecx1
    antenna1.nvecx2 = X@antenna1.nvecx2
    antenna1.nvecy1 = X@antenna1.nvecy1
    antenna1.nvecy2 = X@antenna1.nvecy2
    antenna1.nvecz1 = None # Attached to bus
    antenna1.nvecz2 = None # Blocked by antenna 2
    antenna2.nvecz1 = None # Attached to antenna 1

##### +Y SIDE #######################################################################################
    # rod1 is rotated +45, rod2 is -45, rod3 is +90 (all about Z)
    Z = rot_simple(45, 'z')
    rod1L.nvecx2 = Z@rod1L.nvecx2
    rod1L.nvecx1 = None # Blocked
    rod1L.nvecy1 = None #Blocked
    rod1L.nvecy2 = None #Blocked
    Z = rot_simple(-45, 'z')
    rod2L.nvecx1 = Z@rod2L.nvecx1
    rod2L.nvecx2 = None # Blocked
    rod2L.nvecy1 = None #Blocked
    rod2L.nvecy2 = None #Blocked

    rod3L.nvecx1 = None # Blocked
    rod3L.nvecx2 = None # Blocked
    rod3L.nvecy1 = None # Blocked
    rod3L.nvecy2 = None # Blocked

    rod4L.nvecx1 = None # Blocked
    rod4L.nvecy1 = None # Blocked
    rod4L.nvecy2 = None # Blocked

    rod5L.nvecx2 = None # Blocked
    rod5L.nvecy1 = None # Blocked
    rod5L.nvecy2 = None # Blocked

    panelmidL.nvecx1 = None # Blocked
    panelmidL.nvecx2 = None # Blocked
    panelmidL.nvecy1 = None # Blocked
    panelmidL.nvecy2 = None # Blocked

    panelouterL.nvecy2 = None # Blocked

    panelinnerL.nvecy1 = None # Blocked
    panelinnerL.nvecy2 = None # Blocked

    panelupL.nvecx1 = None # Blocked

    paneldownL.nvecx2 = None # Blocked

##### +Y SIDE #######################################################################################
# rod1 is rotated -45, rod2 is 45, rod3 is +90 (all about Z)
    Z = rot_simple(-45, 'z')
    rod1R.nvecx2 = Z@rod1R.nvecx2
    rod1R.nvecx1 = None # Blocked
    rod1R.nvecy1 = None #Blocked
    rod1R.nvecy2 = None #Blocked
    Z = rot_simple(45, 'z')
    rod2R.nvecx1 = Z@rod2R.nvecx1
    rod2R.nvecx2 = None # Blocked
    rod2R.nvecy1 = None #Blocked
    rod2R.nvecy2 = None #Blocked

    rod3R.nvecx1 = None # Blocked
    rod3R.nvecx2 = None # Blocked
    rod3R.nvecy1 = None # Blocked
    rod3R.nvecy2 = None # Blocked

    rod4R.nvecx1 = None # Blocked
    rod4R.nvecy1 = None # Blocked
    rod4R.nvecy2 = None # Blocked

    rod5R.nvecx2 = None # Blocked
    rod5R.nvecy1 = None # Blocked
    rod5R.nvecy2 = None # Blocked

    panelmidR.nvecx1 = None # Blocked
    panelmidR.nvecx2 = None # Blocked
    panelmidR.nvecy1 = None # Blocked
    panelmidR.nvecy2 = None # Blocked

    panelouterR.nvecy1 = None # Blocked

    panelinnerR.nvecy1 = None # Blocked
    panelinnerR.nvecy2 = None # Blocked

    panelupR.nvecx1 = None # Blocked

    paneldownR.nvecx2 = None # Blocked

# Start here