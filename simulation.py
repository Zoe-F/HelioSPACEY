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
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator, griddata

class Simulation:
    """
    This class parses simulation files and stores parameters for time-dependent 
    or time-independent (ie. stationary) simulations, including times and 
    coordinates the data is defined on, as well as variables simulated.
    Simulation data can be accessed and interpolated using the get_data method.
    """
    def __init__(self, sim_file_paths):
        
        if type(sim_file_paths) == str:
            self.file_paths = sim_file_paths.split(',')
            self.file_paths = [path.strip() for path in self.file_paths]
        else:
            self.file_paths = sim_file_paths
        
        # single file = stationary
        self.is_stationary = True if len(self.file_paths) == 1 else False
        
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
                            ' All files must be of the same format.')
        
        self.times = []
        self.times_mjd = []
        self.dt = str
        self.r = []
        self.theta = []
        self.phi = []
        self.variables = {}
        # different file formats might have different naming conventions
        self.allowed_var_names = {'X1': ['X1', 'r', 'radius', 'distance'], 
                                  'X2': ['X2', 'theta', 'lat', 'latitude'], 
                                  'X3': ['X3', 'phi', 'lon', 'longitude'],
                                  'V1': ['V1', 'ur'], 
                                  'V2': ['V2', 'utheta'], 
                                  'V3': ['V3', 'uphi'],
                                  'B1': ['B1', 'br'], 
                                  'B2': ['B2', 'btheta'], 
                                  'B3': ['B3', 'bphi'], 
                                  'BP': ['BP', 'bp'],
                                  'T': ['T', 'temp', 'temperature'], 
                                  'D': ['D', 'rho', 'density'], 
                                  'DP': ['DP', 'dp'],
                                  'Time': ['time', 'Time'], 
                                  'DT': ['DT', 'dt'], 
                                  'NSTEP': ['NSTEP']}
        self._get_times()
        self._get_coordinates()
        self._get_variables()
        
    def __repr__(self):
        return ('MHD Simulation spanning {} days and radii of {} to {} and latitudes of {} to {}'
                .format(self.length.to_value('day'), self.units, self.bodies,
                        self.times[0].iso[0:10], self.times[-1].iso[0:10]))
    
    def __str__(self):
        return ('Timeseries for variable {} with units {} and bodies {} between {} and {}'
                .format(self.variable, self.units, self.bodies,
                        self.times[0].iso[0:10], self.times[-1].iso[0:10]))
    
    # Function to find the index of the closest element in a list to a given value
    def _closest(self, lst, value):
        lst = np.asarray(lst)
        idx = (np.abs(lst - value)).argmin()
        return idx
    
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
    
    # get times simulation data is defined for from files 
    def _get_times(self, mjd=False):
        
        # get time for each file
        for path in self.file_paths:
            if self.file_format == 'cdf':
                file = cdf.CDF(path)
                self.times.append(t.Time(file.attget(attribute='start_time', entry=0).get('Data')))
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
            for v in file.cdf_info().get('zVariables'):
                for var_key, var_names in self.allowed_var_names.items():
                    if v in var_names:
                        var = var_key
                        self.variables[var] = [v]
                try:
                    unit = file.varattsget(variable=v, expand=True).get('units')[0]
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


    def get_data(self, variable, time, interpolate=False):

        if not self.times_mjd:
            self._get_times(mjd=True)
            
        if type(time) == str:
            try:
                time = t.Time(time)
            except ValueError:
                raise Exception('The time format of the input time string is '\
                                'ambiguous. Please input time as an '\
                                'astropy.time.Time quantity or use a '\
                                'non-ambiguous time format (eg. ISO).')
        
        path = self.file_paths[self._closest(self.times_mjd, time.mjd)]
        
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