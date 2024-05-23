# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:46:44 2023

@author: Zoe.Faes
"""

# imports
import numpy as np
import astropy.units as u
import astropy.time as t
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames #HeliocentricInertial, HeliographicCarrington
#import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import dill as pickle
from bisect import bisect_left, bisect_right


from coordinates import Coordinates
from simulation import Simulation

def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the warning category and message
    return '\n{}: {} \n'.format(category.__name__, msg)

warnings.formatwarning = custom_formatwarning # set warning formatting

class Timeseries:
    
    def __init__(self):
        self.spacecraft = []
        self.variables = []
        self.units = {}
        self.data = []
        # keys: 'times', 'r', 'lat', 'lon', *variables, 'spacecraft', 'conjunction'
        
    def __repr__(self):
        return ('Synthetic timeseries for {} between {} and {}, for variables: {}'
                .format(self.sc, self.data.loc[0,'times'], 
                        self.data.loc[-1,'times'], self.variables))
    
    def __str__(self):
        return ('Synthetic timeseries for {} between {} and {}, for variables: {}'
                .format(self.sc, self.data.loc[0,'times'], 
                        self.data.loc[-1,'times'], self.variables))
    
    ###########################  PRIVATE FUNCTIONS  ###########################
            
    def _nearest(self, 
                 lst: list, 
                 value: int | float):
        """
        Find the index of the nearest element in a list to a given value.

        Parameters
        ----------
        lst : list
            List to be searched.
        value : int | float
            Value that list elements are compared to.
            
        Returns
        -------
        idx: int
            Index of element nearest to value.

        """
        
        idx = bisect_left(lst, value)
        if idx == len(lst):
            return idx - 1
        if idx == 0:
            return 0
        if lst[idx] - value < value - lst[idx-1]:
            return idx
        else:
            return idx - 1
        
        
    ###########################  PUBLIC FUNCTIONS  ############################
        
    def query_sim_data(self, 
                       sc: str, 
                       variables: list[str], 
                       sim: Simulation, 
                       coords: Coordinates):
        """
        Query simulation data at chosen spacecraft and for chosen variables 
        over time range inherited from coords object.

        Parameters
        ----------
        sc : str
            Spacecraft to be 'flown' through simulation space. Choose from
            Simulation.spacecraft
        variables : list[str]
            Variables for which data is to be obtained. Choose from 
            Simulation.variables
        sim : heliospacey.simulation.Simulation
            Simulation object from which data is to be queried.
        coords : heliospacey.coordinates.Coordinates
            Coordinates object containing spacecraft coordinates.

        Returns
        -------
        pandas.DataFrame
            DataFrame with times, spacecraft coordinates (r, lat, lon), 
            and chosen variables in subsequent columns. 

        """
        
        # get spacecraft coordinates at times
        sc_coords = np.array([[c.distance.au, c.lat.rad, c.lon.rad] for c in coords.sc_coords[sc]]).transpose()
        r_spline = CubicSpline(coords.times_mjd, sc_coords[0])
        lat_spline = CubicSpline(coords.times_mjd, sc_coords[1])
        lons = sc_coords[2]
        for l, lon in enumerate(lons):
            if lon < 0:
                lons[l] = lon + 2*np.pi
        lon_spline = CubicSpline(coords.times_mjd, coords._loop_longitudes(lons))
        
        # get timeseries
        times = sim.sc_cell_times[sc]
        timeseries = {}
        timeseries['times'] = times
        timeseries['r'] = r_spline(times)
        timeseries['lat'] = lat_spline(times)
        lons = lon_spline(times)
        lons = lons % (2*np.pi)
        for l, lon in enumerate(lons):
            if lon > np.pi:
                lons[l] = lon - 2*np.pi
        timeseries['lon'] = lons
        
        for var in variables:
            values = []
            for idxs in sim.sc_cell_idxs[sc]:
                values.append(sim.get_value(var, idxs))
            timeseries[var] = values
            
        return pd.DataFrame(timeseries)

    
    def get_flow_paths(self, 
                       sim: Simulation, 
                       coords: Coordinates):
        """
        Get simulated solar wind flow paths for each spacecraft in 
        coords.spacecraft and for each time in self.data['times']
        
        Parameters
        ----------
        sim : heliospacey.simulation.Simulation
            Simulation object used to trace the flow paths
        coords : heliospacey.coordinates.Coordinates
            Coordinates object from which spacecraft coordinates are obtained

        Returns
        -------
        flows_times : dict
            Dictionary with coords.spacecraft as keys, containing lists of times
            for each flow path, corresponding to flows_coords
        flows_coords : dict
            Dictionary with coords.spacecraft as keys, containing lists of 
            coordinates for each flow path, corresponding to flows_times

        """
        
        flows_times = {}
        flows_coords = {}
        for sc in coords.spacecraft:
            print("Computing flow paths for {}...".format(sc))
            
            subset = self.data[self.data['spacecraft'] == sc]
            
            flow_times = [] # times [mjd]
            flow_coords = [] # r, theta, phi
            for time, lon, lat, r in zip(subset['times'].values, subset['lon'].values, subset['lat'].values, subset['r'].values):
                # Fix longitude again
                if lon < 0:
                    lon = lon + 2*np.pi
                coord = [r, lat, lon]
                dts, flow_coord, _, _ = sim.trace_flow(coord, max_r = 1.1)
                # dts = time intervals since plasma volume was at spacecraft
                flow_times.append(np.array(dts) + time)
                flow_coords.append(np.array(flow_coord))
            flows_times[sc] = flow_times
            flows_coords[sc] = flow_coords
            
        print("Flow paths acquired. \n")
        
        return flows_times, flows_coords
    
    
    def get_labels(self,
                   sim: Simulation, 
                   coords: Coordinates, 
                   max_search_distance: tuple[u.Quantity] = (5*u.deg, 0.016*u.au, 0.25*u.day)):
        """
        Get timeseries labels corresponding to conjunctions determined from
        the intersection of spacecraft trajectories with simulated solar wind 
        flow paths.

        Parameters
        ----------
        sim : heliospacey.simulation.Simulation
            Simulation object used to determine solar wind flow paths from 
            spacecraft.
        coords : heliospacey.coordinates.Coordinates
            Coordinates object containing spacecraft coordinates.
        max_search_distance : tuple[astropy.units.Quantity], optional
            Parameters for conjunction search between spacecraft position and 
            flow path (max_angular_distance, max_radial_distance, max_time_difference). 
            The default is (5*u.deg, 0.016*u.au, 0.26*u.day).

        Returns
        -------
        labels : list
            List of labels in the following format: [sc1, sc2, time1, time2]

        """

        # Get flow paths from simulation for each spacecraft
        flows_times, flows_coords = self.get_flow_paths(sim, coords)
        
        # Conjunction search parameters
        rad_limit = max_search_distance[0].to_value(u.rad)
        au_limit = max_search_distance[1].to_value(u.au)
        time_limit = max_search_distance[2].to_value(u.day) # mjd time format
        
        # Keep track of label number
        label_count = 0
        # Keep track of conjunction times
        times_test = []
        labels = []
        for sc1 in self.spacecraft:
            
            print("Finding conjunctions for {}...".format(sc1))
            subset = self.data[self.data['spacecraft'] == sc1]
            other_spacecraft = [sc for sc in self.spacecraft if sc != sc1]
            
            for time, lon_sc, lat, r in zip(subset['times'].values, subset['lon'].values, subset['lat'].values, subset['r'].values):
                
                # Longitudes are defined differently for the simulation and for the S/C coordinates 
                if lon_sc < 0:
                    lon = lon_sc + 2*np.pi
                else:
                    lon = lon_sc
                    
                # TODO: restrict search to "nearby" flows, ie. +- 15 days?
                for sc2 in other_spacecraft:
                    for flow_times, flow_coords in zip(flows_times[sc2], flows_coords[sc2]):
                        # For each flow path
                        for flow_time, flow_coord in zip(flow_times, flow_coords):
                            # For each time & coordinate in the flow path
                            # Identify conjunctions
                            if abs(lon - flow_coord[2]) < rad_limit:
                                if abs(lat - flow_coord[1]) < rad_limit:
                                    if abs(r - flow_coord[0]) < au_limit:
                                        if abs(time - flow_time) < time_limit:
                                            labels.append([sc1, sc2, time, flow_times[0]])
                                            # 'time' corresponds to sc1 time
                                            # flow_times[0] corresponds to sc2 time
                                            times_test.append(time)
                                            times_test.append(flow_times[0])
                                            label_count += 1
            
        print("Number of labels: {}".format(label_count))
        print("Latest label: {} \n".format(max(times_test)))
                    
        return labels
    
    
    def get_timeseries(self, 
                       spacecraft_names: list[str] | str, 
                       variables: list[str] | str, 
                       sim: Simulation, 
                       coords: Coordinates, 
                       set_labels: bool = False):
        """
        Get synthetic timeseries for given spacecraft and simulation variables 
        for time range inherited from coords object and save data to csv file.

        Parameters
        ----------
        spacecraft_names : list[str] | str
            Names of chosen spacecraft. Choose from Simulation.spacecraft.
        variables : list[str] | str
            Chosen simulation variables. Choose from Simulation.variables
        sim : heliospacey.simulation.Simulation
            Simulation object from which timeseries data is to be queried.
        coords : heliospacey.coordinates.Coordinates
            Coordinates object from which spacecraft coordinate data is to be 
            queried.
        set_labels : bool, optional
            Set timeseries labels according to conjunctions determined from 
            simulated data. The default is False.

        Returns
        -------
        path : str
            Path to csv file containing timeseries data.

        """
        
        # parse variables
        if type(variables) == str:
            variables = [var.strip() for var in variables.split(',')]
        for v in variables:
            for var_key, var_names in sim.allowed_var_names.items():
                if v in var_names:
                    self.variables.append(var_key)
                    self.units[var_key] = sim.variables.get(var_key)[1]
                    
        self.spacecraft = coords._parse_sc_names(spacecraft_names)
        
        path = './timeseries/{}_timeseries_{}_{}_{}.csv'.format(sim.name,
                '_'.join(self.spacecraft), coords.times[0].iso[0:10], coords.times[-1].iso[0:10])
        
        all_timeseries = []
        for sc in self.spacecraft:
            print("Querying timeseries for {}...".format(sc))
            
            timeseries = self.query_sim_data(sc, self.variables, sim, coords)
            timeseries['spacecraft'] = sc
            
            all_timeseries.append(timeseries)
            
        all_timeseries = pd.concat(all_timeseries, ignore_index=True, sort=False)
        all_timeseries.to_csv(path)
        self.data = all_timeseries
        
        print("Timeseries acquired. \n")
        
        if set_labels:            
            # assign labels
            all_timeseries['conjunction'] = 'none'
            all_timeseries['conjunction_time'] = 'none'
            labels = self.get_labels(sim, coords)
            for label in labels:
                    # Populate new columns for rows satisfying the conditions
                    all_timeseries.loc[((all_timeseries['spacecraft'] == label[0]) & (all_timeseries['times'] == label[2])), 'conjunction'] = label[1]
                    all_timeseries.loc[((all_timeseries['spacecraft'] == label[0]) & (all_timeseries['times'] == label[2])), 'conjunction_time'] = label[3]
                    all_timeseries.loc[((all_timeseries['spacecraft'] == label[1]) & (all_timeseries['times'] == label[3])), 'conjunction'] = label[0]
                    all_timeseries.loc[((all_timeseries['spacecraft'] == label[1]) & (all_timeseries['times'] == label[3])), 'conjunction_time'] = label[2]
            
            all_timeseries.to_csv(path)
            self.data = all_timeseries
        
        print("Timeseries saved to {} \n".format(path))
    
        return path
        
    
    # TODO: update this function
    def get_timeseries_from_conj(self, sim, variable, conj, plot=False):

        sim._get_times(mjd=True)
        sim._get_variables()
        
        data = []
        coords = []
        
        omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)

        for i in range(len(conj.times)):
            data_array, units, _ = sim.get_data(variable, conj.times[i])
            values = []
            coord = []
            
            for j in range(len(conj.spacecraft)):
                if sim.is_stationary:
                    dt = t.TimeDelta(conj.times[i] - sim.times[0]).to_value('hr')
                    dlon = dt*omega.value
                    phi_idx = sim._nearest(sim.phi.value + dlon, 
                                           conj.coords[i][j].lon.rad)
                else:
                    phi_idx = sim._nearest(sim.phi.value, 
                                           conj.coords[i][j].lon.rad)
                theta_idx = sim._nearest(sim.theta.value, 
                                         conj.coords[i][j].lat.rad)
                r_idx = sim._nearest(sim.r.value, 
                                     conj.coords[i][j].distance.au)
                    
                obstime = sim.times[self._nearest(sim.times_mjd, conj.times[i].mjd)]
                
                values.append(data_array[phi_idx, theta_idx, r_idx])
                coord.append(SkyCoord(
                    sim.phi[phi_idx], sim.theta[theta_idx], sim.r[r_idx], 
                    frame = frames.HeliocentricInertial(obstime = obstime))
                    )
            data.append(values)
            coords.append(coord)

        data = np.asarray(data).transpose()
        coords = np.asarray(coords).transpose()
        
        self.variable = variable
        self.units = units
        self.times = conj.times
        self.coords = coords
        for i, sc in enumerate(conj.spacecraft):
            self.data[sc] = data[i]
    
