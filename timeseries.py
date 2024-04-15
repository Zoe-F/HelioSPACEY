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

def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the warning category and message
    return '\n{}: {} \n'.format(category.__name__, msg)

warnings.formatwarning = custom_formatwarning # set warning formatting

class TimeSeries:
    
    def __init__(self):
        self.spacecraft = []
        self.variables = []
        self.units = []
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
            
    # Function to find the index of the closest element in a list to a given value
    def _nearest(self, lst, value):
        idx = bisect_left(lst, value)
        if idx == len(lst):
            return idx - 1
        if idx == 0:
            return 0
        if lst[idx] - value < value - lst[idx-1]:
            return idx
        else:
            return idx - 1
    
    def _parse_sc_names(self, spacecraft_names):
        # parse specified spacecraft names and store in self.spacecraft
        if type(spacecraft_names) == str:
            spacecraft_names = spacecraft_names.split(',')
        
        allowed_names = {'so': ['so', 'solar orbiter', 'solo'], 
                         'psp': ['psp', 'parker solar probe'], 
                         'bepi': ['bepi', 'bepicolombo', 'bepi colombo'], 
                         'sta': ['sa', 'sta', 'stereo-a', 'stereo a', 'stereoa'], 
                         'earth': ['earth', 'erde', 'aarde', 'terre', 'terra', 
                                   'tierra', 'blue dot', 'home', 'sol d']}
        
        spacecraft = []
        for name in spacecraft_names:
            name = name.strip()
            no_match = True
            for key, names in allowed_names.items():
                if name.lower() in names:
                    spacecraft.append(key)
                    no_match = False
            if no_match:
                raise Exception('Invalid spacecraft name. Specify choice with '\
                                'a string containing the name of a spacecraft,'\
                                ' for example: \'Solar Orbiter\' or \'SolO\'.'\
                                ' spacecraft other than Solar Orbiter, Parker'\
                                ' Solar Probe, BepiColombo and STEREO-A are'\
                                ' not yet supported -- got \'{}\''
                                .format(name))
                    
        self.spacecraft = sorted(spacecraft)
    
    def get_timeseries(self, spacecraft_names, variables, sim, coords, set_labels=False):
        """
        Query synthetic timeseries for chosen spacecraft and specified 
        simulation variables for given time interval and write data to .csv file
        
        Parameters
        ----------
        spacecraft_names : 'list'
            List of allowed spacecraft names. 
            Choose from Simulation.spacecraft
        time_interval : 'list'
            List of strings, datetime or astropy.Time objects.
            Interval over which timeseries is queried: [start_time, end_time]
        variables : 'list'
            List of strings of simulation variables.
            Choose from Simulation.variables
        sim : '~simulation.Simulation'
            Simulation object 
        coords: '~coordinates.Coordinates'
            Coordinates object

        Returns
        -------
        'str'
            Path to .csv file with the following naming convention: 
            timeseries_[spacecraft_names]_[time_interval].csv

        """
        # parse variables
        if type(variables) == str:
            variables = [var.strip() for var in variables.split(',')]
        for v in variables:
            for var_key, var_names in sim.allowed_var_names.items():
                if v in var_names:
                    self.variables.append(var_key)
                    self.units.append(sim.variables.get(var_key)[1])
                    
        self._parse_sc_names(spacecraft_names)
        
        path = './timeseries/{}_timeseries_{}_{}_{}.csv'.format(sim.name,
                '_'.join(self.spacecraft), coords.times[0].iso[0:10], coords.times[-1].iso[0:10])
        
        all_timeseries = []
        for sc in self.spacecraft:
            print("Querying timeseries for {}".format(sc))
            
            timeseries = self.query_sim_data(sc, self.variables, sim, coords)
            timeseries['spacecraft'] = sc
            
            all_timeseries.append(timeseries)
            
        all_timeseries = pd.concat(all_timeseries, ignore_index=True, sort=False)
        all_timeseries.to_csv(path)
        self.data = all_timeseries
        
        if set_labels:
            # get labels
            labels = self.get_labels(sim, coords)
            # assign labels
            all_timeseries['conjunction'] = 'none'
            all_timeseries['conjunction_time'] = 'none'
            for sc in self.spacecraft:
                for label in labels[sc]:
                    # Populate the new column for rows satisfying the condition
                    all_timeseries.loc[((all_timeseries['times'] == label[0]) & (all_timeseries['spacecraft'] == sc)), 'conjunction'] = label[2]
                    all_timeseries.loc[((all_timeseries['times'] == label[0]) & (all_timeseries['spacecraft'] == sc)), 'conjunction_time'] = label[1]
            all_timeseries.to_csv(path)
            self.data = all_timeseries
        
        print("Timeseries saved to {}".format(path))
    
        return path
        
    
    def query_sim_data(self, sc, variables, sim, coords):
        """
        Query synthetic timeseries for a chosen spacecraft from a Simulation
        object
        
        Parameters
        ----------
        sc : 'str'
            Spacecraft name. Choose from Simulation.spacecraft
        time_interval : 'list'
            List of strings, datetime or astropy.Time objects.
            Interval over which timeseries is queried: [start_time, end_time]
        variables : 'list'
            Simulation variable for the timeseries. Choose from Simulation.variables
        sim : '~simulation.Simulation'
            Simulation object from which data is queried
        
        Returns
        -------
        '~pandas.DataFrame'
            DataFrame with times in the first column, and chosen variables in 
            subsequent columns.            
        
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
        
        # TODO: figure out how to deal with longitude range change
        
        # get timeseries
        times = sim.sc_cell_times[sc]
        timeseries = {}
        timeseries['times'] = times
        timeseries['r'] = r_spline(times)
        timeseries['lat'] = lat_spline(times)
        lons = lon_spline(times)
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
    

    def get_labels(self, sim, coords, max_search_radius = [5*u.deg, 0.016*u.au]):
        # in: V, sc_cell_idxs
        # out: self.data[label] = labels
        
        flows_times = {}
        flows_coords = {}
        for sc in self.spacecraft:
            
            print("Computing flow paths for {}...".format(sc))
            
            subset = self.data[self.data['spacecraft'] == sc]
            
            flow_times = [] # times [mjd]
            flow_coords = [] # r, theta, phi
            for time, lon, lat, r in zip(subset['times'].values, subset['lon'].values, subset['lat'].values, subset['r'].values):
                if lon < 0:
                    lon = lon + 2*np.pi
                coord = [r, lat, lon]
                dts, flow_coord, _, _ = sim.trace_flow(coord, max_r = 1.1)
                flow_times.append(np.array(dts) + time)
                flow_coords.append(np.array(flow_coord))
            flows_times[sc] = flow_times
            flows_coords[sc] = flow_coords
            
            # # quick save flow paths for each spacecraft in case of interruption
            # with open('./restart/flow_paths_{}.pickle'.format('_'.join(flow_times.keys())), 'wb') as file:
            #     # save data
            #     pickle.dump([flow_times, flow_idxs], file)
            
        print("Flow paths acquired.")
        
        rad_limit = max_search_radius[0].to_value(u.rad)
        au_limit = max_search_radius[1].to_value(u.au)
        
        labels = {}
        for sc1 in self.spacecraft:
            print("Finding conjunctions for {}...".format(sc1))
            label = []
            other_spacecraft = [sc for sc in self.spacecraft if sc != sc1]
            for time, lon, lat, r in zip(self.data['times'].values, self.data['lon'].values, self.data['lat'].values, self.data['r'].values):
                for sc2 in other_spacecraft:
                    for flow_times, flow_coords in zip(flows_times[sc2], flows_coords[sc2]):
                        for flow_time, flow_coord in zip(flow_times, flow_coords):
                            if abs(lon - flow_coord[2]) < rad_limit:
                                if abs(lat - flow_coord[1]) < rad_limit:
                                    if abs(r - flow_coord[0]) < au_limit:
                                        if abs(time - flow_time) < 0.26:
                                            label.append([time, flow_times[0], sc2])
            labels[sc1] = label
            
        # labels = {}
        # for sc1 in self.spacecraft:
        #     print("Finding conjunctions for {}...".format(sc1))
        #     label = []
            
        #     # Get spacecraft other than sc1
        #     other_spacecraft = [sc for sc in self.spacecraft if sc != sc1]
            
        #     # Iterate over rows of self.data
        #     for idx, row in self.data.iterrows():
        #         time, lon, lat, r = row['times'], row['lon'], row['lat'], row['r']
                
        #         # Iterate over other spacecraft
        #         for sc2 in other_spacecraft:
        #             for flow_times, flow_coords in zip(flows_times[sc2], flows_coords[sc2]):
                        
        #                 # Calculate differences with tolerances
        #                 lon_diff = np.abs(lon - flow_coords[:, 2])
        #                 lat_diff = np.abs(lat - flow_coords[:, 1])
        #                 r_diff = np.abs(r - flow_coords[:, 0])
        #                 time_diff = np.abs(time - flow_times)
                        
        #                 # Check if any match within tolerances
        #                 mask = (lon_diff <= rad_limit) & (lat_diff <= rad_limit) & \
        #                        (r_diff <= au_limit) & (time_diff <= 0.26)
                        
        #                 # Add matching points to label
        #                 if np.any(mask):
        #                     match_idx = np.argmax(mask)
        #                     match_time = flow_times[match_idx]
        #                     label.append([time, match_time, sc2])
        #                     break  # Break the loop if a match is found
            
        #     # Store labels for sc1
        #     labels[sc1] = label
                    
        return labels
        
        
        # TODO: update this function
    def get_timeseries_from_conj(self, sim, variable, conj, plot=False):

        sim._get_times(mjd=True)
        sim._get_variables()
        
        data = []
        coords = []
        
        omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)

        for i in range(len(conj.times)):
            data_array, units = sim.get_data(variable, conj.times[i])
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
    
    # TODO: move to features
    def get_lag(self, resamp_factor=2, check_lag_plots=False):
        
        lags = []
        pcoefs = []
        ignore_duplicates = []
        
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                if not (i == j) and not (j in ignore_duplicates):
                    x = []
                    for time in self.times:
                        x.append(time.jd)
                    spline1 = CubicSpline(x, self.data[i])
                    spline2 = CubicSpline(x, self.data[j])
                    # double sample points in timeseries for lag identification
                    xs = np.arange(
                        x[0], 2*x[-1]-x[-2], (x[-1]-x[0])/(resamp_factor*len(x))
                        )
                    c = np.correlate(spline1(xs)-np.mean(self.data[i]), 
                                     spline2(xs)-np.mean(self.data[j]), 
                                     'full')
    
                    if len(c)%2 == 0:
                        lag = np.argmax(c)-len(c)/2
                    else:
                        lag = np.argmax(c)-(len(c)+1)/2
                    if lag < 0:
                        synced_data1 = spline1(xs[:int(lag)])
                        synced_data2 = spline2(xs[int(-lag):])
                    elif lag == 0:
                        synced_data1 = spline1(xs)
                        synced_data2 = spline2(xs)
                    else:
                        synced_data1 = spline1(xs[int(lag):])
                        synced_data2 = spline2(xs[:int(-lag)])
                        
                    if check_lag_plots:
                        
                        plt.rcParams.update({'text.usetex': True, 
                                             'font.family': 'Computer Modern Roman'})
                        
                        # Plotting options
                        sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
                        colors = ['indianred', 'darkgreen', 'slategrey', 
                                  'steelblue', 'sandybrown', 'slategrey']
                        labels = ['BepiColombo', 'Earth', 'PSP', 
                                  'Solar Orbiter', 'STEREO-A']
                        
                        # FIG 1: cross-correlation plot - max gives lag value
                        title = ('Cross-correlation of timeseries for {} and {}'
                                 .format(labels[sc_index.get(self.spacecraft[i])],
                                         labels[sc_index.get(self.spacecraft[j])]))
                        
                        fig1 = plt.figure(figsize=(8,8), dpi=300)
                        ax = fig1.add_subplot()
                        xc = np.linspace(-len(c)/2, len(c)/2, len(c))
                        ax.plot(xc, c)
                        ax.set_title(title, pad=10, fontsize = 'x-large')
                        
                        # FIG 2: synchronised timeseries - features should overlap
                        title = 'Synchronized timeseries at {}'.format(
                            ', '.join([labels[sc_index.get(sc)] for sc in self.spacecraft])
                            )
                        
                        fig2 = plt.figure(figsize=(8,8), dpi=300)
                        ax = fig2.add_subplot()
                        ax.set_ylabel('{} [{}]'.format(self.variable, self.units), 
                                      fontsize='large')
                        
                        xs_step_in_hours = t.TimeDelta(xs[1]-xs[0], format='jd').to_value('hr')
                        xs1 = np.arange(0, len(xs))
                        xs2 = xs1.copy() + lag
                        ax.plot(xs1*xs_step_in_hours, spline1(xs), 
                                color=colors[sc_index.get(self.spacecraft[0])], 
                                label=labels[sc_index.get(self.spacecraft[0])])
                        ax.plot(xs2*xs_step_in_hours, spline2(xs), 
                                color=colors[sc_index.get(self.spacecraft[1])], 
                                label=labels[sc_index.get(self.spacecraft[1])])
                        
                        ax.set_title(title, pad=45, fontsize = 'x-large')
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                                  ncol=len(self.spacecraft), frameon=False, fontsize='large')
                        ax.set_xlabel(r'$\mathrm{Duration \: [hours]}$', 
                                      fontsize='x-large', labelpad=10)
    
                    lags.append(lag*self.dt.to_value('hr'))
                    pcoefs.append(np.corrcoef(synced_data1, synced_data2))
            ignore_duplicates.append(i)
    
        return lags, pcoefs
    
    # TODO: move to features
    def get_expected_lag(self, conj, sim, coord, check_expected_plot=True):
        
        sim._get_times(mjd=True)
        
        lag = []
        for idx in range(len(conj.times)-1):
            
            V1, units = sim.get_data('V1', conj.times[idx])
            V2, units = sim.get_data('V2', conj.times[idx])
            V3, units = sim.get_data('V3', conj.times[idx])
            
            r = []; theta = []; phi = []; sw_vel = []
            
            for i, coord in enumerate(conj.coords[idx]): 
                
                r.append(coord.distance.au)
                theta.append(coord.lat.rad)
                phi.append(coord.lon.rad)
                
                if sim.is_stationary:
                    omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)
                    dt = t.TimeDelta(conj.times[idx] - sim.times[0]).to_value('hr')
                    dlon = dt*omega.value
                    phi_idx = self._nearest(sim.phi.value + dlon, phi[i])
                else:
                    phi_idx = self._nearest(sim.phi.value, phi[i])
                theta_idx = self._nearest(sim.theta.value, theta[i])
                r_idx = self._nearest(sim.r.value, r[i])
                
                sw_vel.append(
                    [(V1[phi_idx, theta_idx, r_idx]*units).to_value(u.au/u.hr), 
                     (V2[phi_idx, theta_idx, r_idx]*units).to_value(u.au/u.hr), 
                     (V3[phi_idx, theta_idx, r_idx]*units).to_value(u.au/u.hr)]
                    )
                
            x = [r[1]-r[0], theta[1]-theta[0], phi[1]-phi[0]]
            X = np.sqrt(r[0]**2 + r[1]**2 - 2 * r[0] * r[1] * (
                np.sin(theta[0]) * np.sin(theta[1]) * np.cos(phi[0]-phi[1]) + 
                np.cos(theta[0]) * np.cos(theta[1])
                ))
            v = sw_vel[0] if r[0] <= r[1] else sw_vel[1]
            x_unit = x/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            Vx = v[0] * x_unit[0] * (
                np.sin(v[1]) * np.sin(x_unit[1]) * np.cos(v[2]-x_unit[2]) + 
                np.cos(v[1]) * np.cos(x_unit[1])
                )
            lag.append(X/Vx)
        lag.append(X/Vx)
    
        meanlag = np.mean(lag)
        minlag = min(lag)
        maxlag = max(lag)
        
        if check_expected_plot:
            
            title = 'Expected lag from solar wind speed and distance between spacecraft'
            
            fig3 = plt.figure(figsize=(8,8), dpi=300)
            ax = fig3.add_subplot()
            
            duration = range(len(conj.times))*conj.dt.to_value('hr')
            ax.plot(duration, lag, color = 'steelblue')
            ax.fill_between(duration, y1 = np.percentile(lag, 10), 
                            y2 = np.percentile(lag, 90), 
                            color = 'lightsteelblue', alpha=0.5)
            ax.plot(duration, [meanlag]*len(conj.times), 
                    linestyle = '--', color='slategrey')
            
            ax.set_xlabel('Conjunction duration [hrs]', 
                          fontsize='x-large', labelpad=10)
            ax.set_ylabel('Expected lag [hrs]', fontsize='large')
            ax.set_title(title, pad=10, fontsize = 'x-large')
            ax.set_xlim([duration[0], duration[-1]])
            
        return meanlag, minlag, maxlag
    
