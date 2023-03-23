# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:21:20 2023

@author: Zoe.Faes
"""

# imports
import numpy as np
import astrospice
import astropy.units as u
import astropy.time as t
from sunpy.coordinates import frames
import warnings

def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the warning category and message
    return '\n{}: {} \n'.format(category.__name__, msg)

warnings.formatwarning = custom_formatwarning # set warning formatting

class Coordinates:
    """
    This class finds and stores coordinates for the specified spacecraft at 
    the specified times from SPICE kernels. Coordinates can be given in a 
    chosen coordinate system using Sunpy's coordinates.frames objects.
    """
    #############################  CONSTRUCTORS  #############################
    
    def __init__(self, start_time = 'auto', end_time = 'auto', dt = 'auto',
                 spacecraft_names = ['Solar Orbiter', 'PSP', 'Earth'], sim = None):
    
    # initialises Coordinates class by parsing the requested spacecraft names,
    # automatically detecting required start and end times - as well as
    # timestep - if a simulation file is passed, and downloading the required
    # SPICE kernels.
        
        print('Initialising coordinates...')
        
        if start_time == 'auto' and end_time == 'auto' and not sim:
            raise Exception('If no simulation files are provided, start_time'\
                            ' and end_time must be specified.')
        
        elif start_time != 'auto' and end_time != 'auto' and not sim:
            
            warnings.warn('No simulation files provided. All methods depending'\
                          ' on simulation input will be unavailable.')
            # need at least two bodies to find conjunctions: only functionality 
            # available without a simulation file
            if len(spacecraft_names) < 2:
                raise Exception('Please specify the names of at least two spacecraft.')
                
            self.start_time = t.Time(start_time)
            self.end_time = t.Time(end_time)
            if dt == 'auto':
                self.dt = t.TimeDelta(12*u.hour)
                warnings.warn('Timestep \'dt\' is set to 12 hours by default.')
            else:
                self.dt = t.TimeDelta(dt)
            self.times = t.Time(np.arange(self.start_time, self.end_time, self.dt))
            
        elif sim:
            
            if start_time == 'auto' and end_time == 'auto':
                if sim.is_stationary:
                    raise Exception('If simulation is time-independent, '\
                                    'start_time and end_time must be specified.')
                sim._get_times()
                self.times = sim.times
                self.start_time = self.times[0]
                self.end_time = self.times[-1]
                if dt == 'auto':
                    self.dt = sim.dt
                    print('Automatically detected timestep \'dt\' is {:.2f} hours'
                          .format(self.dt.to_value('hr')))
                else:
                    self.dt = t.TimeDelta(dt)
            else:
                self.start_time = t.Time(start_time)
                self.end_time = t.Time(end_time)
                if dt == 'auto':
                    if sim.is_stationary:
                        self.dt = t.TimeDelta(12*u.hour)
                        warnings.warn('For time-independent simulations, dt '\
                                      'is set to 12 hours by default.')
                    else:
                        self.dt = sim.dt
                else:
                    self.dt = t.TimeDelta(dt)
                self.times = t.Time(np.arange(self.start_time, self.end_time, self.dt))
            
        # create coordinate system objects to be set with _set_coordinate_system
        self.coordinate_system = str
        self.coordinate_name = str
        
        # parse specified spacecraft names and store in self.bodies
        if type(spacecraft_names) == str:
            spacecraft_names = spacecraft_names.split(',')
        
        allowed_names = {'so': ['so', 'solar orbiter', 'solo'], 
                         'psp': ['psp', 'parker solar probe'], 
                         'bepi': ['bepi', 'bepicolombo', 'bepi colombo'], 
                         'sta': ['sa', 'sta', 'stereo-a', 'stereo a', 'stereoa'], 
                         'earth': ['earth', 'erde', 'aarde', 'terre', 'terra', 
                                   'tierra', 'blue dot', 'home', 'sol d']}
        
        bodies = []
        for name in spacecraft_names:
            name = name.strip()
            no_match = True
            for key, names in allowed_names.items():
                if name.lower() in names:
                    bodies.append(key)
                    no_match = False
            if no_match:
                raise Exception('Invalid spacecraft name. Specify choice with '\
                                'a string containing the name of a spacecraft,'\
                                ' for example: \'Solar Orbiter\' or \'SolO\'.'\
                                ' spacecraft other than Solar Orbiter, Parker'\
                                ' Solar Probe, BepiColombo and STEREO-A are'\
                                ' not yet supported -- got \'{}\''
                                .format(name))
                    
        self.bodies = sorted(bodies)
        
        print('Getting SPICE kernels...')
        
        # Get kernels corresponding to specified bodies
        if 'so' in self.bodies:
            so_kernels = astrospice.registry.get_kernels('solar orbiter', 'predict')
            self.so_kernel = so_kernels[0]
            so_coverage = self.so_kernel.coverage('SOLAR ORBITER')
            if so_coverage[0] > self.start_time or so_coverage[1] < self.end_time:
                warnings.warn('The Solar Orbiter spice kernel covers the '\
                              'following time range: {} to {}. Please change'\
                              ' the selected time range or remove Solar '\
                              'Orbiter from your selection of spacecraft.'
                              .format(str(so_coverage[0].iso), str(so_coverage[1].iso)))
        
        if 'psp' in self.bodies:
            psp_kernels = astrospice.registry.get_kernels('psp', 'predict')
            self.psp_kernel = psp_kernels[0]
            psp_coverage = self.psp_kernel.coverage('SOLAR PROBE PLUS')
            if psp_coverage[0] > self.start_time or psp_coverage[1] < self.end_time:
                warnings.warn('The Parker Solar Probe spice kernel covers the'\
                              'following time range: {} to {}. Please change '\
                              'the selected time range to not exceed this '\
                              'coverage or remove Parker Solar Probe from '\
                              'your selection of spacecraft.'
                              .format(str(psp_coverage[0].iso), str(psp_coverage[1].iso)))
            
        if 'bepi' in self.bodies:
            try:
                self.bepi_kernel = astrospice.kernel.SPKKernel(
                    './SPK/bc_mpo_fcp_00129_20181020_20251101_v0.bsp')
            except:
                raise Exception('bc_mpo_fcp_00129_20181020_20251101_v01.bsp '\
                                'could not be found. Please download the '\
                                'file from https://repos.cosmos.esa.int/socci'\
                                '/projects/SPICE_KERNELS/repos/bepicolombo/'\
                                'browse/kernels/spk and ensure it is '\
                                'accessible by the current filepath: ./SPK/')
            bepi_coverage = self.bepi_kernel.coverage('BEPICOLOMBO MPO')
            if bepi_coverage[0] > self.start_time or bepi_coverage[1] < self.end_time:
                warnings.warn('The BepiColombo spice kernel covers the '\
                              'following time range: {} to {}. Please change'\
                              ' the selected time range to not exceed this '\
                              'coverage or remove BepiColombo from your '\
                              'selection of spacecraft.'
                              .format(str(bepi_coverage[0].iso), str(bepi_coverage[1].iso)))
        
        if 'sta' in self.bodies:
            sta_kernels = astrospice.registry.get_kernels('stereo-a', 'predict')
            self.sta_kernel = sta_kernels[0]
            sta_coverage = self.sta_kernel.coverage('STEREO AHEAD')
            if sta_coverage[0] > self.start_time or sta_coverage[1] < self.end_time:
                warnings.warn('The STEREO-A spice kernel covers the following'\
                              ' time range: {} to {}. Please change the '\
                              'selected time range to not exceed this '\
                              'coverage or remove STEREO-A from your '\
                              'selection of spacecraft.'
                              .format(str(sta_coverage[0].iso), str(sta_coverage[1].iso)))
            
        if 'earth' in self.bodies:
            try:
                self.planets_kernel = astrospice.kernel.SPKKernel('./SPK/de430.bsp')
            except:
                raise Exception('de430.bsp could not be located. Please '\
                                'download the file from https://naif.jpl.nasa'\
                                '.gov/pub/naif/generic_kernels/spk/planets/ '\
                                'and ensure it is accessible by the current '\
                                'filepath: ./SPK/')
            earth_coverage = self.planets_kernel.coverage('EARTH')
            if earth_coverage[0] > self.start_time or earth_coverage[1] < self.end_time:
                warnings.warn('The Earth spice kernel covers the following '\
                              'time range: {} to {}. Please change the '\
                              'selected time range to not exceed this '\
                              'coverage or remove Earth from your selection '\
                              'of spacecraft.'
                              .format(str(bepi_coverage[0].iso), str(bepi_coverage[1].iso)))

        
    def __repr__(self):
        return ('There are {} conjunctions between {} from {} to {}'
                .format(self.nallconjs, ', '.join(self.bodies), self.start_time, self.end_time))
    
    def __str__(self):
        return ('There are {} conjunctions between {} from {} to {}'
                .format(self.nallconjs, ', '.join(self.bodies), self.start_time, self.end_time))


    ##########################  PRIVATE FUNCTIONS  ###########################

    # returns the index of the closest element in a list to a given value
    def _closest(self, lst, value):
        lst = np.asarray(lst)
        idx = (np.abs(lst - value)).argmin()
        return idx   
    
    # converts cartesian coordinates to spherical coordinates in radians
    # output: r, lat, lon
    def _cartesian2spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)
        return r, theta, phi

    # converts spherical coordinates in radians to cartesian coordinates
    # input: r, lat, lon
    def _spherical2cartesian(self, r, theta, phi):
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return x, y, z

    # sets coordinate system for which coordinates will be given.
    # input: sunpy.coordinates.frames object, or str for HCI or HCG coordinates
    def _set_coordinate_system(self, coordinate_system):
        
        # lists of allowed names for each coordinate system
        allowed_HCI_names = ['hci', 
                             'heliocentricinertial', 
                             'heliocentric inertial']
        allowed_HGC_names = ['hgc', 
                             'heliographiccarrington', 
                             'heliographic carrington', 
                             'carrington']
        
        if type(coordinate_system) == str:
            # select chosen coordinate system
            if coordinate_system.lower() in allowed_HCI_names:
                self.coordinate_system = frames.HeliocentricInertial()
                self.coordinate_name = 'Heliocentric Inertial'
            elif coordinate_system.lower() in allowed_HGC_names:
                self.coordinate_system = frames.HeliographicCarrington(observer='earth')
                self.coordinate_name = 'Heliographic Carrington'
            else:
                raise Exception('Please enter a valid string for HCI or HGC '\
                                'frames, or alternatively, a '\
                                'sunpy.coordinates.frames object.')
        
        
        elif getattr(coordinate_system, '__module__', None) == frames.__name__:
            self.coordinate_system = coordinate_system
            # catch HCI and HCG frame objects for nicer formatting in plots
            if getattr(coordinate_system, 'name', None) in allowed_HCI_names:
                self.coordinate_name = 'Heliocentric Inertial'
            elif getattr(coordinate_system, 'name', None) in allowed_HGC_names:
                self.coordinate_name = 'Heliographic Carrington'
            else:
                self.coordinate_name = getattr(coordinate_system, 'name', None)
        
        else:
            raise Exception('Please enter a valid string or a '\
                            'sunpy.coordinates.frames object.')

    # gets coordinates in a specified coordinate system from SPICE kernels for 
    # a set of bodies for given times. Currently supported bodies:
    # Solar Orbiter, Parker Solar Probe, BepiColombo, STEREO-A and Earth.
    def get_coordinates(self, bodies, times, coordinate_system=frames.HeliocentricInertial()):

        self._set_coordinate_system(coordinate_system)

        kernel_keys = {'bepi': 'BEPICOLOMBO MPO', 
                       'earth': 'EARTH', 
                       'psp': 'SOLAR PROBE PLUS',
                       'so': 'SOLAR ORBITER', 
                       'sta': 'STEREO AHEAD'}
        
        coords = []
        for i in range(len(bodies)):
            # Get coordinates for times
            coords.append(astrospice.generate_coords(kernel_keys.get(bodies[i]), times))
            # Transform to chosen frame
            coords[i] = coords[i].transform_to(self.coordinate_system)
        
        return coords
    
    # calculates instantaneous velocity of object at each time based on list 
    # of coordinates corresponding to those times
    def get_instantaneous_velocity(self, coords, cartesian_output=True):
        
        velocities = []
        if cartesian_output:
            units = [u.au/u.hr, u.au/u.hr, u.au/u.hr]
            for coord in np.asarray(coords).transpose():
                vel = []
                for i in range(len(coord)-1):
                    # get time difference
                    dt = t.TimeDelta(coord[i+1].obstime - coord[i].obstime).to_value('hr')
                    # convert to cartesian coordinats
                    x1, y1, z1 = self._spherical2cartesian(coord[i].distance.au, 
                                                           coord[i].lat.rad, 
                                                           coord[i].lon.rad)
                    x2, y2, z2 = self._spherical2cartesian(coord[i+1].distance.au, 
                                                           coord[i+1].lat.rad, 
                                                           coord[i+1].lon.rad)
                    # get velocity components
                    xdot = (x2-x1)/dt
                    ydot = (y2-y1)/dt
                    zdot = (z2-z1)/dt
                    vel.append([xdot, ydot, zdot])
                # assign velocity for last time to be velocity of time t-1
                vel.append([xdot, ydot, zdot])
                velocities.append(vel)
        else:
            units = [u.au/u.hr, u.rad/u.hr, u.rad/u.hr]
            for coord in np.asarray(coords).transpose():
                vel = []
                for i in range(len(coord)-1):
                    # get time difference
                    dt = t.TimeDelta(coord[i+1].obstime - coord[i].obstime).to_value('hr')
                    # get velocity in spherical coordinates
                    r = coord[i].distance.au
                    rdot = (coord[i+1].distance.au - r)/dt
                    thetadot = (coord[i+1].lat.rad - coord[i].lat.rad)/dt
                    phidot = (coord[i+1].lon.rad - coord[i].lon.rad)/dt
                    v = [rdot, r*np.sin(coord[i].lon.rad)*thetadot, r*phidot]
                    vel.append(v)
                # assign velocity for last time to be velocity of time t-1
                vel.append(v)
                velocities.append(vel)
        
        return velocities, units
    
    
# TODO: would be nice to add an add_spice_kernel() function to get coordinates 
#       from a local unsupported spk file