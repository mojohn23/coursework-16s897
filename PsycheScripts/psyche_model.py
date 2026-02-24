import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

density = 98.82 # [kg/m^3]

class BoxObj:
    def __init__(self, height: float, width: float, depth: float):
        self.height = height # [m]
        self.width = width # [m]
        self.depth = depth # [m]
        self.volume = height*width*depth # [m^3]
        self.com = np.transpose(np.array([depth/2, width/2, height/2], ndmin = 2)) # [m]
        Ixx = 1/12*(width**2 + height**2)
        Iyy = 1/12*(depth**2 + height**2)
        Izz = 1/12*(depth**2 + width**2)
        self.sMOICOM = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.mass = density*self.volume # [kg]
        self.MOICOM = self.mass*self.sMOICOM

class FrustumObj:
    def __init__(self, r1: float, r2: float, height: float):
        self.height = height # [m]
        self.r1 = r1 # [m], note that: r1 > r2
        self.r2 = r2 # [m]
        self.volume = 1/3*m.pi*height*(r1**2 + r2**2 + (r1*r2))
        self.com = np.transpose(np.array([0, 0, (height*((r1 + r2)**2 + 2*r2**2))/(4*((r1 + r2)**2 - r1*r2))], ndmin = 2)) # [m]
        Ixx = 1 # Placeholder
        Iyy = Ixx
        Izz = 3/10*(r1**5 - r2**5)/(r1**3 - r2**3)
        self.sMOICOM = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.mass = density*self.volume # [kg]
        self.MOICOM = self.mass*self.sMOICOM
        # slant = m.sqrt((r1 - r2)**2 + height**2)
        self.SA = None # [m^2], placeholder

class RodObj:
    def __init__(self, radius:float, length:float):
        self.radius = radius # [m]
        self.length = length # [m]
        self.volume = m.pi*radius**2*length # [m^3]
        self.com = np.transpose(np.array([0, length/2, 0], ndmin = 2)) # [m]
        Ixx = 1/12*length**2
        Iyy = 0
        Izz = 1/12*length**2
        self.sMOICOM = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.mass = density*self.volume # [kg]
        self.MOICOM = self.mass*self.sMOICOM

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



if __name__ == '__main__':
    ##### MAIN BODY ######################################################################################
    # Bus
    bus = BoxObj(2.9, 2.2, 2.4)
    bus.com = np.transpose(np.array([0, 0, 0], ndmin = 2)) # Assume bus COM is the origin

    # Antenna bottom, rotated 180 and placed on body +Z face
    antenna1 = FrustumObj(2, 1, 0.5)
    antenna1.com = np.transpose(np.array([0, 0, 2.9/2 + 0.5], ndmin = 2)) - antenna1.com

    # Antenna top, placed on top of antenna bottom
    antenna2 = FrustumObj(2, 0.2, 0.25)
    antenna2.com = antenna2.com + np.transpose(np.array([0, 0, 2.9/2 + 0.5], ndmin = 2))

    ##### SUPPORTS +Y #################################################################################
    # Rod1 rotated +45 and placed along body +Y face
    rod1L = RodObj(0.025, 1.6829)
    Z = rot_simple(45, 'z')
    rod1L.com = Z@rod1L.com + np.transpose(np.array([0, 2.2/2, 0], ndmin = 2))

    # Rod2 rotated -45 and placed along body +Y face
    rod2L = RodObj(0.025, 1.6829)
    Z = rot_simple(-45, 'z')
    rod2L.com = Z@rod2L.com + np.transpose(np.array([0, 2.2/2, 0], ndmin = 2))

    # Rod3 rotated 90 and placed at the tip of rod1 and rod2
    rod3L = RodObj(0.025, 2.4)
    rod3L.com = np.transpose(np.array([0, 2.2/2 + 1.2, 0], ndmin = 2))

    # Rod4 placed at the tip of rod1
    rod4L = RodObj(0.025, 0.55)
    rod4L.com = rod4L.com + np.transpose(np.array([-2.4/2, 2.2/2 + 1.2, 0], ndmin = 2))

    # Rod5 placed at the tip of rod3
    rod5L = RodObj(0.025, 0.55)
    rod5L.com = rod5L.com + np.transpose(np.array([2.4/2, 2.2/2 + 1.2, 0], ndmin = 2))

    ##### SOLAR PANELS +Y ##############################################################################
    panelmidL = BoxObj(0.1, 3.18, 2.4)
    panelmidL.com = np.transpose(np.array([0, 1.74 + 3.18 + 3.18/2, 0], ndmin = 2))
    panelleftL = BoxObj(0.1, 3.18, 2.4)
    panelleftL.com = np.transpose(np.array([0, 1.74 + 2*3.18 + 3.18/2, 0], ndmin = 2))
    panelrightL = BoxObj(0.1, 3.18, 2.4)
    panelrightL.com = np.transpose(np.array([0, 1.74 + 3.18/2, 0], ndmin = 2))
    panelupL = BoxObj(0.1, 3.18, 2.4)
    panelupL.com = np.transpose(np.array([-2.4, 1.74 + 3.18 + 3.18/2, 0], ndmin = 2))
    paneldownL = BoxObj(0.1, 3.18, 2.4)
    paneldownL.com = np.transpose(np.array([2.4, 1.74 + 3.18 + 3.18/2, 0], ndmin = 2))
    
    ##### SUPPORTS -Y #################################################################################
    # Rod1 rotated -45 and placed along body -Y face
    rod1R = RodObj(0.025, 1.6829)
    Z = rot_simple(-45, 'z')
    rod1R.com = Z@rod1R.com + np.transpose(np.array([0, -2.2/2, 0], ndmin = 2))

    # Rod2 rotated 45 and placed along body -Y face
    rod2R = RodObj(0.025, 1.6829)
    Z = rot_simple(45, 'z')
    rod2R.com = Z@rod2R.com + np.transpose(np.array([0, -2.2/2, 0], ndmin = 2))

    # Rod3 rotated 90 and placed at the tip of rod1 and rod2
    rod3R = RodObj(0.025, 2.4)
    rod3R.com = np.transpose(np.array([0, -2.2/2 - 1.2, 0], ndmin = 2))

    # Rod4 placed at the tip of rod1
    rod4R = RodObj(0.025, 0.55)
    rod4R.com = rod4R.com + np.transpose(np.array([-2.4/2, -2.2/2 - 1.2, 0], ndmin = 2))

    # Rod5 placed at the tip of rod3
    rod5R = RodObj(0.025, 0.55)
    rod5R.com = rod5R.com + np.transpose(np.array([2.4/2, -2.2/2 - 1.2, 0], ndmin = 2))

    ##### SOLAR PANELS -Y #############################################################################
    panelmidR = BoxObj(0.1, 3.18, 2.4)
    panelmidR.com = np.transpose(np.array([0, -(1.74 + 3.18 + 3.18/2), 0], ndmin = 2))
    panelleftR = BoxObj(0.1, 3.18, 2.4)
    panelleftR.com = np.transpose(np.array([0, -(1.74 + 2*3.18 + 3.18/2), 0], ndmin = 2))
    panelrightR = BoxObj(0.1, 3.18, 2.4)
    panelrightR.com = np.transpose(np.array([0, -(1.74 + 3.18/2), 0], ndmin = 2))
    panelupR = BoxObj(0.1, 3.18, 2.4)
    panelupR.com = np.transpose(np.array([-2.4, -(1.74 + 3.18 + 3.18/2), 0], ndmin = 2))
    paneldownR = BoxObj(0.1, 3.18, 2.4)
    paneldownR.com = np.transpose(np.array([2.4, -(1.74 + 3.18 + 3.18/2), 0], ndmin = 2))

    ## Full spacecraft
    tot_bus_vol = bus.volume + antenna1.volume + antenna2.volume
    tot_left_vol = rod1L.volume + rod2L.volume + rod3L.volume + rod4L.volume + rod5L.volume + panelmidL.volume + panelleftL.volume + panelrightL.volume + panelupL.volume + paneldownL.volume
    tot_right_vol = rod1R.volume + rod2R.volume + rod3R.volume + rod4R.volume + rod5R.volume + panelmidR.volume + panelleftR.volume + panelrightR.volume + panelupR.volume + paneldownR.volume
    tot_vol = tot_bus_vol + tot_left_vol + tot_right_vol
    tot_mass = density*tot_vol
    # total_mass = bus.mass + antenna1.mass + antenna2.mass + 2*(rod1L.mass + rod2L.mass + rod3L.mass + rod4L.mass + rod5L.mass + 5*panelmidL.mass)
    
    tot_bus_com = bus.mass*bus.com + antenna1.mass*antenna1.com + antenna2.mass*antenna2.com
    tot_left_com = rod1L.mass*rod1L.com + rod2L.mass*rod2L.com + rod3L.mass*rod3L.com + rod4L.mass*rod4L.com + rod5L.mass*rod5L.com + panelmidL.mass*panelmidL.com + panelleftL.mass*panelleftL.com + panelrightL.mass*panelrightL.com + panelupL.mass*panelupL.com + paneldownL.mass*paneldownL.com
    tot_right_com = rod1R.mass*rod1R.com + rod2R.mass*rod2R.com + rod3R.mass*rod3R.com + rod4R.mass*rod4R.com + rod5R.mass*rod5R.com + panelmidR.mass*panelmidR.com + panelleftR.mass*panelleftR.com + panelrightR.mass*panelrightR.com + panelupR.mass*panelupR.com + paneldownR.mass*paneldownR.com
    tot_com = (tot_bus_com + tot_left_com + tot_right_com)/tot_mass

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
    # ax.plot(panelleftL.com[0], panelleftL.com[1], panelleftL.com[2], 'go')
    # ax.plot(panelrightL.com[0], panelrightL.com[1], panelrightL.com[2], 'go')
    # ax.plot(panelupL.com[0], panelupL.com[1], panelupL.com[2], 'go')
    # ax.plot(paneldownL.com[0], paneldownL.com[1], paneldownL.com[2], 'go')

    # ax.plot(rod1R.com[0], rod1R.com[1], rod1R.com[2], 'ro')
    # ax.plot(rod2R.com[0], rod2R.com[1], rod2R.com[2], 'ro')
    # ax.plot(rod3R.com[0], rod3R.com[1], rod3R.com[2], 'ro')
    # ax.plot(rod4R.com[0], rod4R.com[1], rod4R.com[2], 'ro')
    # ax.plot(rod5R.com[0], rod5R.com[1], rod5R.com[2], 'ro')

    # ax.plot(panelmidR.com[0], panelmidR.com[1], panelmidR.com[2], 'go')
    # ax.plot(panelleftR.com[0], panelleftR.com[1], panelleftR.com[2], 'go')
    # ax.plot(panelrightR.com[0], panelrightR.com[1], panelrightR.com[2], 'go')
    # ax.plot(panelupR.com[0], panelupR.com[1], panelupR.com[2], 'go')
    # ax.plot(paneldownR.com[0], paneldownR.com[1], paneldownR.com[2], 'go')

    # ax.set_ylim(-10, 10)
    # ax.set_xlim(-10, 10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()