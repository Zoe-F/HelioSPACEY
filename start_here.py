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

# heliospacey imports
from coordinates import Coordinates
from conjunction import Conjunctions
from simulation import Simulation
from timeseries import Timeseries
import plotter as plot


######################  GETTING SPACECRAFT COORDINATES  #######################

# Specify times
times = np.arange(t.Time("2022-01-01 00:00:00.000"), 
                  t.Time("2022-03-01 00:00:00.000"),
                  t.TimeDelta(6*u.hour))

# Specify spacecraft
spacecraft = ['bepi', 'earth', 'psp', 'so', 'sta']

# Instantiate coordinates
coords = Coordinates(times, spacecraft)

# Position of Solar Orbiter at noon on Pi day 2022
time = t.Time("2022-03-14 12:00:00.000")
so_coords = coords.get_sc_coordinates('solar orbiter', time)[0]
print("\n Solar Orbiter position: {} \n".format(so_coords))

# Plot spacecraft positions
plot.spacecraft_position(coords, coords.spacecraft, time, print_coordinates=False)


##########################  GETTING SIMULATION DATA  ##########################

# Instantiate simulation
sim_file_path = "./simulations/Zoe_Faes_101922_SH_1/helio/3D_CDF/helio.enlil.0000.cdf"
sim = Simulation(sim_file_path)

# Find radial velocity (V1) at the Solar Orbiter coordinates
coordinates = [so_coords.distance, so_coords.lat + 90*u.deg, so_coords.lon] # sim space latitudes range from 30 to 150
data, units, value = sim.get_data('V1', coordinates = coordinates)
print("\n Radial velocity at coordinates: {} {}".format(value, units))

# Plot simulation slices
plot.ENLIL_slice(sim, 'V1', lat=coordinates[1])
plot.ENLIL_slice(sim, 'V1', lon=coordinates[2])
plot.ENLIL_slice(sim, 'V1', radius=coordinates[0])


#############  WORKING WITH VIRTUAL SPACECRAFT IN THE SIMULATION  #############

# Important if working with spacecraft coordinates and simulation together:
# Find corresponding spacecraft coordinates in simulation space
for sc in coords.spacecraft:
    sim.get_cells_from_sc_coords(coords, sc)
    
# Check simulation cells and spacecraft coordinates are consistent
# plot.sc_sim_consistency(coords, sim, spacecraft=sim.spacecraft)

# Instantiate timeseries
ts = Timeseries()

# Get timeseries
spacecraft = sim.spacecraft # all available spacecraft in sim
variables = list(sim.variables.keys())[3:] # all available variables in sim
ts.get_timeseries(spacecraft, variables, sim, coords, set_labels=False)

# plot timeseries
plot.timeseries(ts, times, variables, 'solar orbiter, psp')

# # generate timeseries animation - WARNING: SLOW
# plot.timeseries_reference('solar orbiter', 'V1', [times[0], times[-1]], ts, sim)

# # Trace solar wind flows from spacecraft
# flows_times, flows_coords = ts.get_flow_paths(sim, coords)

# # Animate flow tracing - WARNING: SLOW
# plot.flow_from_spacecraft(coords, 'psp', sim_file_path, cmap='plasma')

# # Plot the last flow path from psp
# psp_flow_path = flows_coords.get('psp')[-1:]
# plot.vector_field(sim, 'V', background_var='B3', flow_paths=psp_flow_path)


######################  FINDING GEOMETRIC CONJUNCTIONS  #######################

# Instantiate conjunctions
conj = Conjunctions(coords.times, coords.spacecraft, coords.sc_coords, 
                    sc_cell_idxs = sim.sc_cell_idxs, 
                    sc_cell_times = sim.sc_cell_times)

# Predict conjunctions
conj.get_all_conjunctions(sim)
print(conj)

# Conjunctions can be quickly accessed by type with the following attributes
# print(conj.cones, conj.quads, conj.opps, conj.parkers)

# Find conjunctions by type, spacecraft and/or time
# Conjunction types to choose from: cone, quadrature, opposition or parker spiral
so_psp_conj = conj.find_conjunctions(spacecraft_names=['solar orbiter', 'psp'])
parker_psp_conj = conj.find_conjunctions(conj_type = 'parker spiral', spacecraft_names='psp')
new_year_conj = conj.find_conjunctions(time='2022-01-01 00:00:00.000')

# Look at the new year conjunctions
print(new_year_conj)

# Plot conjunction
plot.conjunction_2D(coords, conj, parker_psp_conj[0])

# Plot conjunction with background radial velocity slice
plot.conjunction_2D(coords, conj, so_psp_conj[0], ENLIL=True, sim=sim, variable='V1')

