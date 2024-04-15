# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:26:09 2023

@author: Zoe.Faes
"""

# imports
import numpy as np
import astropy.units as u
import astropy.time as t
import cdflib as cdf
import netCDF4 as nc
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator, griddata, CubicSpline
from bisect import bisect_left, bisect_right
import os

from coordinates import Coordinates

class Simulation:
    """
    This class parses simulation files and stores parameters for time-dependent 
    or time-independent (ie. stationary) simulations, including times and 
    coordinates the data is defined on, as well as variables available.
    """
    def __init__(self, sim_file_paths):
        
        # simulation attributes
        self.file_paths = []
        self.name = str
        self.file_format = str
        self.is_stationary = bool
        self.times = []
        self.times_mjd = []
        self.dt = str
        self.r = []
        self.theta = []
        self.phi = []
        self.phi0 = float # phase difference in longitude at times[0] for stationary simulations in radians
        self.omega = (2*np.pi*u.rad/(25.38*u.day).to(u.s)).value # Carrington rotation rate [rad/s]
        self.au_to_m = (1*u.au).to_value(u.m)
        self.variables = {}
        self.data = {}
        
        # spacecraft-simulation correspondence
        self.spacecraft = []
        self.sc_cell_times = {}
        self.sc_cell_idxs = {}
        
        # accommodate different naming conventions
        self.allowed_var_names = {'X1': ['X1', 'x1', 'r', 'radius', 'distance'], 
                                  'X2': ['X2', 'x2', 'theta', 'lat', 'latitude'], 
                                  'X3': ['X3', 'x3', 'phi', 'lon', 'longitude'],
                                  'V1': ['V1', 'v1', 'ur'], 
                                  'V2': ['V2', 'v2', 'utheta'], 
                                  'V3': ['V3', 'v3', 'uphi'],
                                  'B1': ['B1', 'b1', 'br'], 
                                  'B2': ['B2', 'b2', 'btheta'], 
                                  'B3': ['B3', 'b3', 'bphi'], 
                                  'BP': ['BP', 'bp'],
                                  'T': ['T', 't', 'temp', 'temperature'], 
                                  'D': ['D', 'd', 'rho', 'density'], 
                                  'DP': ['DP', 'dp'],
                                  'Time': ['time', 'Time'], 
                                  'DT': ['DT', 'dt'], 
                                  'NSTEP': ['NSTEP', 'nstep']}
        
        self._parse_file_paths(sim_file_paths)
        
        # single-file simulation is assumed to be stationary
        self.is_stationary = True if len(self.file_paths) == 1 else False
        
        self._get_times()
        self._get_coordinates()
        self._get_variables()
        self._store_data()
        
    def __repr__(self):
        return ('{} ENLIL simulation {}'.format(
            'Stationary' if self.is_stationary else 'Time-dependent', self.name))
    
    # def __str__(self):
    #     return ('Timeseries for variable {} with units {} and spacecraft {} between {} and {}'
    #             .format(self.variable, self.units, self.spacecraft,
    #                     self.times[0].iso[0:10], self.times[-1].iso[0:10]))
    
    
##############################################################################
#                              PRIVATE FUNCTIONS                             #
##############################################################################
    
    
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
    
    def _min_distance(self, lst, value):
        lst = np.asarray(lst)
        dist = min(np.abs(lst - value))
        return dist
    
    def _spherical_distance(self,
                           coord1: list[float] | np.ndarray[float], 
                           coord2: list[float] | np.ndarray[float]) -> float:
        """
        Computes the Euclidean distance between two points given in 
        spherical coordinates.

        Parameters
        ----------
        coord1 : list[float] | np.ndarray[float]
            [r1, theta1, phi1]
        coord2 : list[float] | np.ndarray[float]
            [r2, theta2, phi2]

        Returns
        -------
        float
            Euclidean distance between the two points.

        """
        
        r1, theta1, phi1 = coord1
        r2, theta2, phi2 = coord2
        dist = np.sqrt(r1**2 + r2**2 - 2*r1*r2 * (np.sin(theta1) * np.sin(theta2) * np.cos(phi1-phi2) + np.cos(theta1) * np.cos(theta2)))
        return dist

    
    # converts spherical coordinates in radians to cartesian coordinates
    # input: r, lat, lon
    def _spherical2cartesian(self, r, theta, phi):
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return x, y, z
    
    # interpolation method 1 - UNTESTED
    def _interpolate(self, r, theta, phi, data, resamp_factor=2):
        rs = np.linspace(min(r), max(r), resamp_factor*len(r))
        thetas = np.linspace(min(theta), max(theta), resamp_factor*len(theta))
        phis = np.linspace(min(phi), max(phi), resamp_factor*len(phi))
        xs, ys, zs = self._spherical2cartesian(rs, thetas, phis)
        X, Y, Z = np.meshgrid(xs, ys, zs)
        x, y, z = self._spherical2cartesian(r, theta, phi)
        interp = LinearNDInterpolator(list(zip(x, y, z)), data)
        interp_data = interp(X, Y, Z)
        return interp_data
    
    # interpolation method 2 - UNTESTED
    def _interpolate2(self, r, theta, phi, data, resamp_factor=2):
        rs = np.linspace(min(r), max(r), resamp_factor*len(r))
        thetas = np.linspace(min(theta), max(theta), resamp_factor*len(theta))
        phis = np.linspace(min(phi), max(phi), resamp_factor*len(phi))
        xs, ys, zs = self._spherical2cartesian(rs, thetas, phis)
        X, Y, Z = np.meshgrid(xs, ys, zs)
        x, y, z = self._spherical2cartesian(r, theta, phi)
        interp_data = griddata((x, y, z), data, (X, Y, Z), method='linear')
        return interp_data
    
    def _parse_file_paths(self, sim_file_paths):
        
        if isinstance(sim_file_paths, str): 
            # parse string
            self.file_paths = sim_file_paths.split(',')
            self.file_paths = [path.strip() for path in self.file_paths]
        elif isinstance(sim_file_paths, list) and isinstance(sim_file_paths[0], str):
            self.file_paths = sim_file_paths
        else:
            raise Exception('sim_file_paths must either be a string or '\
                            'a list of strings.')
        
        # determine file format
        if all([path[-3:] == 'cdf' for path in self.file_paths]):
            self.file_format = 'cdf'
        elif all([path[-2:] == 'nc' for path in self.file_paths]):
            self.file_format = 'nc'
        elif all([path[-4:] == 'fits' for path in self.file_paths]):
            self.file_format = 'fits'
            print('FITS files are only partially supported. Certain methods '\
                  'may not function properly.')
        else:
            self.file_format = None
            raise Exception('Supported file formats are CDF, netCDF and FITS.'\
                            ' All files must have the same format.')
                
        dir_names = os.path.normpath(self.file_paths[0]).split(os.sep)
        idx = dir_names.index('helio') - 1
        self.name = dir_names[idx]
    
    # get times simulation data is defined for from files 
    def _get_times(self, mjd=True):
        
        # get time for each file
        for path in self.file_paths:
            if self.file_format == 'cdf':
                file = cdf.CDF(path)
                self.times.append(t.Time(file.attget(attribute='start_time', entry=0).Data))
            elif self.file_format == 'nc':
                file = nc.Dataset(path)
                time = (t.Time(file.__dict__['crstart_cal']) + 
                        t.TimeDelta(float(file.variables.get('TIME')[0].data) << u.s))
                self.times.append(time)
            elif self.file_format == 'fits':
                with fits.open(path) as hdul:
                    self.times.append(t.Time(hdul[0].header['start_time']))
        
        if self.is_stationary:
            # if sim is stationary, there is no timestep by definition
            self.dt = None
        else:
            # timesteps may not all be exactly equal within double precision 
            dts = [self.times[i+1] - self.times[i] for i in range(len(self.times)-1)]
            self.dt = t.TimeDelta(np.mean(dts))
        
        if mjd:
            for time in self.times:
                self.times_mjd.append(time.mjd)
                
    def _get_coordinates(self):
        # ---- physics convention used here: (r, theta, phi) 
        # ie. theta = lat and phi = lon
        
        if self.file_format == 'cdf':
            file = cdf.CDF(self.file_paths[0])
            # get r, theta, phi coordinates data are defined on
            self.r = (file.varget(variable='r')[0] << u.m).to(u.au)
            self.theta = file.varget(variable='theta')[0] << u.rad
            self.phi = file.varget(variable='phi')[0] << u.rad
            # phis = file.varget(variable='phi')[0]
            # for p, phi in enumerate(phis):
            #     if phi > np.pi:
            #         phis[p] = (phi - 2*np.pi)
            # self.phi = phis << u.rad
            
        elif self.file_format == 'nc':
            file = nc.Dataset(self.file_paths[0])
            self.r = (file['X1'][0] << u.m).to(u.au)
            self.theta = file['X2'][0] << u.rad
            self.phi = file['X3'][0] << u.rad
            
        else:
            raise Exception('File format not supported.')
            
    def _get_variables(self):
        
        if self.file_format == 'cdf':
            file = cdf.CDF(self.file_paths[0])
            # get variable names and units
            for v in file.cdf_info().zVariables:
                for var_key, var_names in self.allowed_var_names.items():
                    if v in var_names:
                        var = var_key
                        self.variables[var] = [v]
                try:
                    unit = file.varattsget(variable=v).get('units')
                except:
                    try:
                        self.variables[var].append(u.Unit(''))
                    except KeyError:
                        self.variables[var] = [v, u.Unit('')]
                try:
                    self.variables[var].append(u.Unit(unit))
                except KeyError:
                    self.variables[var] = [v.name, u.Unit(unit)]
                except ValueError:
                    raise Exception('Unit {} could not be parsed.'.format(unit))
        
        elif self.file_format == 'nc':
            file = nc.Dataset(self.file_paths[0])
            # get variable names and units
            for v in file.variables.values():
                for var_key, var_names in self.allowed_var_names.items():
                    if v.name in var_names:
                        var = var_key
                        self.variables[var] = [v.name]
                try:
                    unit = v.getncattr('units')
                except AttributeError:
                    try:
                        self.variables[var].append(u.Unit(''))
                    except KeyError:
                        self.variables[var] = [v.name, u.Unit('')]
                try:
                    self.variables[var].append(u.Unit(unit))
                except KeyError:
                    self.variables[var] = [v.name, u.Unit(unit)]
                except ValueError:
                    raise Exception('Unit {} could not be parsed.'.format(unit))

        else:
            raise Exception('File format not supported.')
    

    def _store_data(self):
        path = self.file_paths[0]
        data = {}
        if self.file_format == 'cdf':
            for var in list(self.variables.keys())[3:]:
                data[var] = cdf.CDF(path).varget(variable=self.variables.get(var)[0])[0]
                data[var] = np.reshape(data[var], (180,60,320))
            self.data = data
        elif self.file_format == 'nc':
            for var in list(self.variables.keys())[3:]:
                data[var] = nc.Dataset(path)[self.variables.get(var)[0]][0]
            self.data = data
        else:
            raise Exception('File format not supported.')


    def _find_optimal_timestep(self, 
                               r_spline: CubicSpline, 
                               theta_spline: CubicSpline, 
                               phi_spline: CubicSpline, 
                               times_mjd: list[float], 
                               psp: bool = False) -> t.TimeDelta:
        
        # When PSP is at perihelion, its velocity relative to the sun surface
        # is so great that it appears to be moving with retrograde motion,
        # such that dphi_dt becomes negative. To find optimal sampling rate,
        # we need to screen out perihelion times. Also, PSP goes beyond the 
        # inner boundary of the simulation space (0.14 AU)
        if psp:           
            perihelion_mask = [r_spline(time) <= 0.14 for time in times_mjd]
            # find all dphi/dt outside of perihelion intervals
            dr = np.diff(np.ma.masked_array(r_spline(times_mjd), mask=perihelion_mask))
            dtheta = np.diff(np.ma.masked_array(theta_spline(times_mjd), mask=perihelion_mask))
            dphi = np.diff(np.ma.masked_array(phi_spline(times_mjd), mask=perihelion_mask))
            dt = np.diff(np.ma.masked_array(times_mjd, mask=perihelion_mask))
        
        else:
            dr = np.diff(r_spline(times_mjd))
            dtheta = np.diff(theta_spline(times_mjd))
            dphi = np.diff(phi_spline(times_mjd))
            dt = np.diff(times_mjd)
        
        dr_dt = abs(dr)/dt
        dtheta_dt = abs(dtheta)/dt
        dphi_dt = abs(dphi)/dt
        
        # find timestep from max velocity
        r_dt = np.diff(self.r.value).mean()/max(dr_dt)
        theta_dt = np.diff(self.theta.value).mean()/max(dtheta_dt)
        phi_dt = np.diff(self.phi.value).mean()/max(dphi_dt)
        
        return min(r_dt, theta_dt, phi_dt)
        

    def _find_nearest_cells(self, 
                             r_spline: CubicSpline, 
                             theta_spline: CubicSpline, 
                             phi_spline: CubicSpline, 
                             times_mjd: np.ndarray[float], 
                             psp: bool = False,
                             fixed_dt: float | None = None):
        
        # in frame of stationary solution (solar rotation has already been 
        # accounted for in parent function)
        if fixed_dt:
            dt = fixed_dt # in mjd
        else:
            dt = self._find_optimal_timestep(r_spline, theta_spline, phi_spline, times_mjd, psp=psp)
        
        times = np.arange(times_mjd[0], times_mjd[-1], dt)
        
        rs = r_spline(times)
        thetas = theta_spline(times)
        phis = phi_spline(times)
        # account for looping of longitudes
        phis = phis % (2*np.pi)
                
        # TODO: check that returned times reflect time of closest approach to 
        # cell center, not just time at which closest cell is queried
        r_idx = [self._nearest(self.r.value, r) for r in rs]
        theta_idx = [self._nearest(self.theta.value, theta) for theta in thetas]
        phi_idx = [self._nearest(self.phi.value, phi) for phi in phis]
        
        # for i, phi in enumerate(phis):
        #     phis[i] = (phi - 2*np.pi) if (phi > np.pi) else phi
        
        idxs = np.array([r_idx, theta_idx, phi_idx]).transpose()
        distances = [self._spherical_distance([r, theta, phi], idx) for r, theta, phi, idx in zip(rs, thetas, phis, idxs)]
        
        return idxs, times, distances

    
    def _get_flattened_idx(self, i, j, k, R=320, C=60, D=180):
        return i + R*j + R*C*k
        
    
##############################################################################
#                               PUBLIC FUNCTIONS                             #
##############################################################################   

    def get_cells_from_sc_coords(self, coords: Coordinates, spacecraft: str, fixed_dt: float | None = None):
        """
        Find cells and times in simulation-space corresponding to spacecraft 
        coordinates.
        
        Parameters
        ----------
        coords : 'heliospacey.coordinates.Coordinates'
            Coordinates object used to find nearest simulation-space 
            coordinates to the spacecraft path
        spacecraft: 'str'
            Allowed spacecraft name. Choose from Coordinates.spacecraft

        Returns
        -------
        'heliospacey.simulation.Simulation.sc_cell_idxs' :
            indices of simulation coordinates closest to spacecraft coordinates
        'heliospacey.simulation.Simulation.sc_cell_times' : 
            times corresponding to spacecraft's closest approach to simulation points
        """
        # Offset in longitude between t0 for spacecraft coordinates and t0 for simulation coordinates
        phi0 = (self.omega*t.TimeDelta(coords.times[0]-self.times[0]).to_value(u.s)) % (2*np.pi)
        self.phi0 = phi0 << (u.rad)
        
        # get spacecraft coordinates
        r_sc = []
        theta_sc = []
        phi_sc = []
        for i, c in enumerate(coords.sc_coords[spacecraft]):
            r_sc.append(c.distance.au)
            theta_sc.append(c.lat.rad + np.pi/2)
            if c.lon.rad < 0:
                phi_sc.append((c.lon.rad + 2*np.pi - self.omega*t.TimeDelta(coords.times[i]-coords.times[0]).to_value(u.s)) % (2*np.pi))
            else:
                phi_sc.append((c.lon.rad - self.omega*t.TimeDelta(coords.times[i]-coords.times[0]).to_value(u.s)) % (2*np.pi))

                
        # "loop" longitudes to avoid discontinuities at 0 -> pi so spline 
        # doesn't suffer from Gibbs phenomenon
        phi = coords._loop_longitudes(phi_sc)
        
        # fit splines to spacecraft coordinates
        r_spline = CubicSpline(coords.times_mjd, r_sc) # range: [0.14, 2.2] AU
        theta_spline = CubicSpline(coords.times_mjd, theta_sc) # range: [0, pi] rad
        phi_spline = CubicSpline(coords.times_mjd, phi) # range: [-pi, pi] rad
        
        # phi spline residuals
        # plt.plot(range(len(phi)), phi - phi_spline(coords.times_mjd))
        
        # find indices of sim coords closest to spacecraft
        if spacecraft == 'psp':
            idxs, times, distances = self._find_nearest_cells(r_spline, theta_spline, phi_spline, coords.times_mjd, psp=True, fixed_dt=fixed_dt)
        else:
            idxs, times, distances = self._find_nearest_cells(r_spline, theta_spline, phi_spline, coords.times_mjd, fixed_dt=fixed_dt)

        self.sc_cell_times[spacecraft] = times
        self.sc_cell_idxs[spacecraft] = idxs
        self.spacecraft.append(spacecraft)
        
    
    def get_value_slow(self, variable, cell_idxs):
        path = self.file_paths[0]
        if self.file_format == 'cdf':
            idx = self._get_flattened_idx(*cell_idxs)
            value = cdf.CDF(path).varget(variable=self.variables.get(variable)[0])[0][idx]
        elif self.file_format == 'nc':
            value = nc.Dataset(path)[self.variables.get(variable)[0]][0][cell_idxs[0]][cell_idxs[1]][cell_idxs[2]]
        else:
            raise Exception('File format not supported.')
        
        return value
    
    def get_value(self, variable, cell_idxs):
        # phi_idx, theta_idx, r_idx
        value = self.data.get(variable)[cell_idxs[2]][cell_idxs[1]][cell_idxs[0]]
        
        return value
    
    def get_value_time_dep(self, variable, cell_idxs, time=None):
            
        if isinstance(time, str) or isinstance(time, float):
            try:
                time = t.Time(time)
            except ValueError:
                raise Exception('The time format of the input time string is '\
                                'ambiguous. Please input time as an '\
                                'astropy.time.Time quantity or use a '\
                                'non-ambiguous time format (eg. ISO).')
        
        if self.is_stationary:
            path = self.file_paths[0]
        else:
            path = self.file_paths[self._nearest(self.times_mjd, time.mjd)]
        
        if self.file_format == 'cdf':
            idx = self._get_flattened_idx(*cell_idxs)
            value = cdf.CDF(path).varget(variable=self.variables.get(variable)[0])[0][idx]
        elif self.file_format == 'nc':
            value = nc.Dataset(path)[self.variables.get(variable)[0]][0][cell_idxs[0]][cell_idxs[1]][cell_idxs[2]]
        else:
            raise Exception('File format not supported.')
        
        return value

    def get_data(self, variable, time=None, coords=None, interpolate=False):
        
        if not self.is_stationary and time == None:
            raise Exception("Please specify a time for time-dependent simulations.")
            
        elif not self.is_stationary and time != None:                
            if type(time) == str:
                try:
                    time = t.Time(time)
                except ValueError:
                    raise Exception('The time format of the input time string is '\
                                    'ambiguous. Please input time as an '\
                                    'astropy.time.Time quantity or use a '\
                                    'non-ambiguous time format (eg. ISO).')
            
            path = self.file_paths[self._nearest(self.times_mjd, time.mjd)]
        
        else:
            path = self.file_paths[0]
        
        no_match = True
        for key, values in self.allowed_var_names.items():
            if variable in values:
                no_match = False
                if self.file_format == 'cdf':
                    data = cdf.CDF(path).varget(variable=self.variables.get(key)[0])[0]
                    data = np.reshape(data, (180,60,320))
                    units = self.variables.get(key)[1]
                elif self.file_format == 'nc':
                    data = nc.Dataset(path)[self.variables.get(key)[0]][0]
                    units = self.variables.get(key)[1]
                else:
                    raise Exception('File format not supported.')
                if interpolate:
                    data = self._interpolate(self.r, self.theta, self.phi, data)
                    
        if no_match:
            raise Exception('Variable name could not be parsed. Please choose from: {}'
                            .format(self.variables.keys()))
        
        return data, units
    
            
    def find_nearest_cell(self, coord, velocity):
        
        omega = 2*np.pi*u.rad/(25.38*u.day).to(u.s)
        
        # find optimal sampling rate based on solar wind velocity through sim cells
        r_dt = np.diff(self.r.to_value(u.m)).mean() / velocity[0] # dt in seconds
        # arclength = r*angle
        # dtheta = dphi = 0.034906585 rad
        theta_dt = coord[0].to_value(u.m) * np.diff(self.theta.value).mean() / velocity[1]
        phi_dt = coord[0].to_value(u.m) * np.diff(self.phi.value).mean() / velocity[2]
        
        # dt = r * dphi / vphi
        # dphi = dt * vphi / r
        
        dt = min(abs(r_dt), abs(theta_dt), abs(phi_dt))
        
        dr = (velocity[0] * dt) << u.m
        new_r = coord[0] + dr.to(u.au)
        dtheta = velocity[1] * dt / coord[0].to_value(u.m)
        dphi = velocity[2] * dt / coord[0].to_value(u.m)
        new_theta = coord[1] + (dtheta << u.rad)
        new_phi = coord[2] + (dphi << u.rad)
        new_coord = [new_r, new_theta, new_phi]

        r_idx = (self._nearest(self.r.value, new_r.value))
        theta_idx = (self._nearest(self.theta.value, new_theta.value))
        phi_idx = self._nearest(self.phi.value, new_phi.value)
        new_cell_idxs = [r_idx, theta_idx, phi_idx]
        
        dt = t.TimeDelta(dt << u.s).jd
        
        return new_cell_idxs, new_coord, dt
    
    
    def find_nearest_cell_rotated(self, coord, velocity):
        
        # find optimal sampling rate based on solar wind velocity through sim cells
        r_dt = np.diff(self.r.to_value(u.m)).mean() / velocity[0] # dt in seconds
        # arclength = r*angle
        # dtheta = dphi = 0.034906585 rad
        theta_dt = coord[0] * self.au_to_m * np.diff(self.theta.value).mean() / velocity[1]
        phi_dt = coord[0] * self.au_to_m * np.diff(self.phi.value).mean() / velocity[2]
        
        # dt = r * dphi / vphi
        # dphi = dt * vphi / r
        
        dt = min(abs(r_dt), abs(theta_dt), abs(phi_dt))
        
        dr = velocity[0] * dt / self.au_to_m
        new_r = coord[0] + dr
        dtheta = velocity[1] * dt / (coord[0] * self.au_to_m)
        dphi = -self.omega * dt + (velocity[2] * dt / (coord[0] * self.au_to_m))
        new_theta = coord[1] + dtheta
        new_phi = coord[2] + dphi
        new_coord = [new_r, new_theta, new_phi]

        r_idx = self._nearest(self.r.value, new_r)
        theta_idx = self._nearest(self.theta.value, new_theta)
        phi_idx = self._nearest(self.phi.value, new_phi)
        new_cell_idxs = [r_idx, theta_idx, phi_idx]
        
        dt = t.TimeDelta(dt << u.s).jd
        
        return new_cell_idxs, new_coord, dt
    
    
    
    def trace_flow(self, coord, max_r = 1.1):
        """
        Traces simulated plasma flows for labelling of timeseries as either 
        conjunction or non-conjunction. Only for stationary sims, assumes no 
        branching. 

        Parameters
        ----------
        coord : list[float]
            list of 3 coordinates: r in au, theta, phi in radians.

        Returns
        -------
        None.

        """
        
        if not self.is_stationary:
            raise Exception('This method does not support time-dependent '\
                            'simulations.')
        
        # assume no branching
        radius = coord[0]
        dts = [0] # time interval since start of flow in mjd
        coords = [coord]
        velocities = []
        r_idx = self._nearest(self.r.value, coord[0])
        theta_idx = self._nearest(self.theta.value, coord[1])
        phi_idx = self._nearest(self.phi.value, coord[2])
        cell_idxs = [r_idx, theta_idx, phi_idx]
        cells = [cell_idxs]
        
        while radius <= max_r:
            # find velocity at coord
            v_r = self.get_value('V1', cell_idxs)
            v_theta = self.get_value('V2', cell_idxs)
            v_phi = self.get_value('V3', cell_idxs)
            v = [v_r, v_theta, v_phi]
            velocities.append(v)
            
            # find next coord
            cell_idxs, coord, dt = self.find_nearest_cell_rotated(coord, v)
            # store info
            coords.append(coord)
            dt = dt + dts[-1]
            dts.append(dt)
            cells.append(cell_idxs)
            
            # update radius value
            radius = coord[0]
            
        # add last velocity element
        v_r = self.get_value('V1', cell_idxs)
        v_theta = self.get_value('V2', cell_idxs)
        v_phi = self.get_value('V3', cell_idxs)
        v = [v_r, v_theta, v_phi]
        velocities.append(v)
        
        flow_path = [dts, coords, cells, velocities]
        
        return flow_path
        

        
    # def find_nearest_sc(self, flow_path):
        
        
        

        