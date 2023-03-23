# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:46:48 2023

@author: Zoe.Faes
"""
#################################  IMPORTS  ###################################

# useful external imports
import numpy as np
import astropy.units as u
import astropy.time as t
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
import dill as pickle
from datetime import datetime
import matplotlib.pyplot as plt

# toolkit imports
from coordinates import Coordinates
from conjunction import Conjunctions
from simulation import Simulation
from timeseries import TimeSeries
import plotter as p


########################  IF PICKLE FILE: START HERE  #########################

filepath = './Timeseries/Zoe_Faes_101922_SH_1_20200701_20250701_6h.pickle'

# load data
with open(filepath, 'rb') as file:
    # file structure: [Coordinates, Conjunctions, Simulation]
    data = pickle.load(file)
    coordinates = data[0]
    conj = data[1]
    sim = data[2]

# examples
print('Available simulation variables to chose from: ')
for vname, var in sim.variables.items():
    print(vname, var[1])
    if vname == 'V1':
        p.plot_ENLIL_slice(sim, vname, sim.times[0], lat=90*u.deg)

for c in conj.cones:
    if c.length > 100*u.day:
        p.plot_timeseries(c.timeseries['V1'])

conjs = conj.find_conjunctions(category='parker spiral', spacecraft_names='solar orbiter, psp')
for c in conjs[:3]:
    p.plot_conjunction2D(coordinates, c, ENLIL=True, sim=sim, variable='V1')
    
# to train a NN, run the nn.py as a script - remove any pickle files you don't have from the 'file_names' list




#######################  IF NO PICKLE FILE: START HERE  #######################

def __init__(start_time = 'auto', end_time = 'auto', dt = 'auto', 
                 satellite_names = ['solar orbiter', 'parker solar probe', 'earth'], 
                 sim_file_paths = None, coordinate_system = frames.HeliocentricInertial()):
        
    print("Initialising VISTA...")
    
    if sim_file_paths:
        sim = Simulation(sim_file_paths)
    else:
        sim = None
    coordinates = Coordinates(start_time, end_time, dt, satellite_names, sim)
    coords = c.get_coordinates(c.bodies, c.times, coordinate_system)
    conj = Conjunctions(c.times, c.bodies, coords)
    
    return coordinates, conj, sim


# sim_path = '.\Simulations\Zoe_Faes_101922_SH_1\helio\3D_CDF\helio.enlil.0000.cdf'
    
# #SO kernel: start_time = '2020-02-11', end_time = '2030-11-20'
# coordinates, conj, sim = __init__(start_time = '2020-07-01 00:00:00.000', 
#                                   end_time = '2021-01-01 00:00:00.000', 
#                                   dt = 6*u.hour, 
#                                   satellite_names = ['solar orbiter', 
#                                                       'parker solar probe', 
#                                                       'earth', 'STEREO-A', 'bepi'], 
#                                   sim_file_paths = sim_path)

# conj.get_all_conjunctions(sim)

##########################  MISCELLANEOUS TESTING  ###########################

