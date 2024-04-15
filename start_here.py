# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:09:52 2024

@author: Zoe.Faes
"""

#################################  IMPORTS  ###################################

# external imports
import numpy as np
import astropy.units as u
import astropy.time as t
import pandas as pd
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from datetime import datetime
import matplotlib.pyplot as plt

# helio-spacey imports
from coordinates import Coordinates
from conjunction import Conjunctions
from simulation import Simulation
from timeseries import TimeSeries
import plotter as plot
import features as feat

########################  FIND SPACECRAFT COORDINATES  ########################

# Specify times
times = np.arange(t.Time("2022-01-01 00:00:00.000"), 
                  t.Time("2022-07-01 00:00:00.000"),
                  t.TimeDelta(6*u.hour))

# Specify spacecraft
spacecraft_names = ['solar orbiter', 'psp', 'stereo a', 'bepi', 'earth']

# Instantiate coordinates object
coords = Coordinates(times, spacecraft_names)
print(coords)

# Position of Solar Orbiter at noon on Pi day 2022
time = t.Time("2022-03-14 12:00:00.000")
so_coords = coords.sc_coords['so'][np.where(coords.times == time)][0]
print("Solar Orbiter position: ", so_coords)
print("Solar Orbiter distance from Sun: ", so_coords.distance.to(u.au))

time = t.Time("2022-01-01 12:00:00.000")
psp_coords = coords.sc_coords['psp'][np.where(coords.times == time)][0]
print("PSP position: ", psp_coords)

# If working with ENLIL simulations, specify path(s) to CDF or netCDF file(s)
sim_file_path = "./simulations/Zoe_Faes_101922_SH_1/helio/3D_CDF/helio.enlil.0000.cdf"
# Instantiate simulation object
sim = Simulation(sim_file_path)
print(sim)

# Let's find the trajectory of the spacecraft in the simulation
for sc in coords.spacecraft:
    sim.get_cells_from_sc_coords(coords, sc)

# Check simulation cells and spacecraft coordinates are consistent
plot.check_sc_sim_consistency(coords, sim, spacecraft=sim.spacecraft)

# Get timeseries for spacecraft
ts = TimeSeries()
ts.get_timeseries(sim.spacecraft, list(sim.variables.keys())[3:], sim, coords, set_labels=False)

# plot timeseries
plot.simple_timeseries(ts, times, ['B1', 'B2', 'B3', 'V1', 'V2', 'V3', 'D', 'T'], ['T', 'T', 'T', 'm/s', 'm/s', 'm/s', 'kg/m3', 'K'], ['so'])

# plot simulation slice
plot.plot_ENLIL_slice(sim, 'V1', times[0], lat=90*u.deg)

# generate animation
plot.timeseries_reference('so', [times[0], times[-1]], 'V1', ts, sim)

# Animate flow tracing
plot_times = np.arange(t.Time("2022-01-01 00:00:00.000"), 
                        t.Time("2022-04-01 00:00:00.000"),
                        t.TimeDelta(3*u.hour))
plot_coords = Coordinates(plot_times, spacecraft_names)
plot.flow_from_spacecraft(plot_coords, 'psp', sim_file_path, cmap='plasma')


# # Find conjunctions
# conj = Conjunctions(coords.times, coords.spacecraft, coords.sc_coords, 
#                     sc_cell_idxs = sim.sc_cell_idxs, 
#                     sc_cell_times = sim.sc_cell_times)

# conj.get_all_conjunctions(sim)


