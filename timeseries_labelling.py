# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:04:24 2024

@author: Zoe.Faes
"""

# external imports
import numpy as np
import astropy.units as u
import astropy.time as t

# heliospacey imports
from coordinates import Coordinates
from conjunction import Conjunctions
from simulation import Simulation
from timeseries import Timeseries
import plotter as plot

# Specify times
times = np.arange(t.Time("2022-01-01 00:00:00.000"), 
                  t.Time("2023-01-01 00:00:00.000"),
                  t.TimeDelta(6*u.hour))

# Specify spacecraft
spacecraft = ['bepi', 'earth', 'psp', 'so', 'sta']

# Instantiate coordinates
coords = Coordinates(times, spacecraft)

# Instantiate simulation
sim_file_path = "./simulations/Zoe_Faes_101922_SH_1/helio/3D_CDF/helio.enlil.0000.cdf"
sim = Simulation(sim_file_path)

# Fly virtual spacecraft through simulation space
for sc in coords.spacecraft:
    sim.get_cells_from_sc_coords(coords, sc)
    
# # Check simulation cells and spacecraft coordinates are consistent
# plot.sc_sim_consistency(coords, sim, spacecraft=sim.spacecraft)

# Instantiate timeseries
ts = Timeseries()

# Get timeseries & set labels
spacecraft = sim.spacecraft # all available spacecraft in sim
variables = list(sim.variables.keys())[3:] # all available variables in sim
ts.get_timeseries(spacecraft, variables, sim, coords, set_labels=True)


# # Animate flow tracing to confirm timeseries labels
# # Change spacecraft as needed
# # WARNING: SLOW. Run it once for all of 2022
# plot.flow_from_spacecraft(coords, 'psp', sim_file_path, cmap='plasma')
