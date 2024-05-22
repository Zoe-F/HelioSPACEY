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
from bisect import bisect_left
import warnings

def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the warning category and message
    return '\n{}: {} \n'.format(category.__name__, msg)

warnings.formatwarning = custom_formatwarning # set warning formatting

class Coordinates:
    """
    This class queries and stores coordinates for the specified 
    spacecraft between the specified times from SPICE kernels.
    """
    #############################  CONSTRUCTORS  #############################
    
    def __init__(self, 
                 times: list | np.ndarray, 
                 spacecraft_names: str | list[str] = ['Solar Orbiter', 'PSP', 'Earth']):
        """
        Initialise Coordinates class with the times and spacecraft for which 
        the coordinates are to be queried.

        Parameters
        ----------
        times : list | np.ndarray
            List or array of times, either as astropy.time.Time objects, or
            as unambiguous strings (eg. not MJD format which cannot be 
            differentiated from JD format).
        spacecraft_names : str | list[str], optional
            String or list of strings of the names of the required spacecraft. 
            The default is ['Solar Orbiter', 'PSP', 'Earth'].

        Returns
        -------
        None.

        """
    
        # attributes
        self.times = []
        self.times_mjd = []
        # self.dt = float # in hours
        self.spacecraft = []
        self.SPICE_kernels = {} # keys: self.spacecraft
        self.sc_coords = {} # keys: self.spacecraft
        self.coordinate_system = str
        
        # print('Initializing coordinates...')
        
        if type(spacecraft_names) == str:
            names = spacecraft_names.split(',')
            spacecraft_names = [name.strip() for name in names]
            
        self.spacecraft = self._parse_sc_names(spacecraft_names)
        
        self._set_times(times)
        
        self._get_SPICE_kernels()
        
        self.set_coordinates()
        
        # print('Initialization complete. \n')
        
    def __repr__(self):
        return ('Coordinates object for {} from {} to {}'.format(
            ', '.join(self.spacecraft), self.times[0], self.times[-1]))
    
    def __str__(self):
        return ('Coordinates for {} from {} to {}'.format(
            ', '.join(self.spacecraft), self.times[0], self.times[-1]))


    ###########################  PRIVATE FUNCTIONS  ###########################

    def _set_times(self, times):
        if not (all(isinstance(time, t.Time) for time in times)):
            try:
                self.times = t.Time(times)
                self.times_mjd = np.array([time.mjd for time in times])
            except ValueError:
                raise Exception('The time format of the input time string is '\
                                'ambiguous. Please input time as an '\
                                'astropy.time.Time quantity or use a '\
                                'non-ambiguous time format (eg. ISO).')
        else:
            self.times = np.array(times)
            self.times_mjd = np.array([time.mjd for time in times])

    def _parse_sc_names(self, 
                        spacecraft_names: list[str] | str):
        """
        Parse specified spacecraft names and store sorted list in 
        Timeseries.spacecraft.

        Parameters
        ----------
        spacecraft_names : list[str] | str
            Names of chosen spacecraft.

        Returns
        -------
        list[str]
            List of ordered allowed spacecraft names.

        """
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

        return sorted(spacecraft)
        
    def _get_SPICE_kernels(self):
        # Get kernels corresponding to specified spacecraft
        if 'so' in self.spacecraft:
            self.SPICE_kernels['so'] = astrospice.registry.get_kernels('solar orbiter', 'predict')[0]
            print('SOLO KERNEL: {}'.format(self.SPICE_kernels['so']))
            coverage = self.SPICE_kernels['so'].coverage('SOLAR ORBITER')
            if coverage[0] > self.times[0] or coverage[1] < self.times[-1]:
                warnings.warn('The Solar Orbiter spice kernel covers the '\
                              'following time range: {} to {}. Please change'\
                              ' the selected time range or remove Solar '\
                              'Orbiter from your selection of spacecraft.'
                              .format(str(coverage[0].iso), str(coverage[1].iso)))
        
        if 'psp' in self.spacecraft:
            try:
                self.SPICE_kernels['psp'] = astrospice.kernel.SPKKernel(
                    './spice_kernels/spp_nom_20180804_20250831_v001.bsp')
            except:
                raise Exception('spp_nom_20180804_20250831_v001.bsp '\
                                'could not be found. Please download the '\
                                'file from https://psp-gateway.jhuapl.edu/website/Ancillary/LongTermEphemerisPredict'\
                                ' and ensure it is accessible by the current '\
                                'filepath: ./spice_kernels/')
            coverage = self.SPICE_kernels['psp'].coverage('SOLAR PROBE PLUS')
            if coverage[0] > self.times[0] or coverage[1] < self.times[-1]:
                warnings.warn('The Parker Solar Probe spice kernel covers the'\
                              'following time range: {} to {}. Please change '\
                              'the selected time range to not exceed this '\
                              'coverage or remove Parker Solar Probe from '\
                              'your selection of spacecraft.'
                              .format(str(coverage[0].iso), str(coverage[1].iso)))
            
        if 'bepi' in self.spacecraft:
            try:
                self.SPICE_kernels['bepi'] = astrospice.kernel.SPKKernel(
                    './spice_kernels/bc_mpo_fcp_00158_20181020_20251101_v01.bsp')
            except:
                raise Exception('bc_mpo_fcp_00158_20181020_20251101_v01.bsp '\
                                'could not be found. Please download the '\
                                'file from https://repos.cosmos.esa.int/socci'\
                                '/projects/SPICE_KERNELS/repos/bepicolombo/'\
                                'browse/kernels/spk and ensure it is '\
                                'accessible by the current filepath: ./spice_kernels/')
            coverage = self.SPICE_kernels['bepi'].coverage('BEPICOLOMBO MPO')
            if coverage[0] > self.times[0] or coverage[1] < self.times[-1]:
                warnings.warn('The BepiColombo spice kernel covers the '\
                              'following time range: {} to {}. Please change'\
                              ' the selected time range to not exceed this '\
                              'coverage or remove BepiColombo from your '\
                              'selection of spacecraft.'
                              .format(str(coverage[0].iso), str(coverage[1].iso)))
        
        if 'sta' in self.spacecraft:
            try:
                self.SPICE_kernels['sta'] = astrospice.kernel.SPKKernel(
                    './spice_kernels/ahead_2017_061_5295day_predict.epm.bsp')
            except:
                raise Exception('ahead_2017_061_5295day_predict.epm.bsp could not be found. '\
                                'Please download the file from https://naif.jpl.nasa.gov/pub/naif/STEREO/kernels/spk/'\
                                ' and ensure it is accessible by the current '\
                                'filepath: ./spice_kernels/')
            coverage = self.SPICE_kernels['sta'].coverage('STEREO AHEAD')
            if coverage[0] > self.times[0] or coverage[1] < self.times[-1]:
                warnings.warn('The STEREO-A spice kernel covers the following'\
                              ' time range: {} to {}. Please change the '\
                              'selected time range to not exceed this '\
                              'coverage or remove STEREO-A from your '\
                              'selection of spacecraft.'
                              .format(str(coverage[0].iso), str(coverage[1].iso)))
            
        if 'earth' in self.spacecraft:
            try:
                self.SPICE_kernels['earth'] = astrospice.kernel.SPKKernel('./spice_kernels/de430.bsp')
            except:
                raise Exception('de430.bsp could not be located. Please '\
                                'download the file from https://naif.jpl.nasa'\
                                '.gov/pub/naif/generic_kernels/spk/planets/ '\
                                'and ensure it is accessible by the current '\
                                'filepath: ./spice_kernels/')
            coverage = self.SPICE_kernels['earth'].coverage('EARTH')
            if coverage[0] > self.times[0] or coverage[1] < self.times[-1]:
                warnings.warn('The Earth spice kernel covers the following '\
                              'time range: {} to {}. Please change the '\
                              'selected time range to not exceed this '\
                              'coverage or remove Earth from your selection '\
                              'of spacecraft.'
                              .format(str(coverage[0].iso), str(coverage[1].iso)))


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
            elif coordinate_system.lower() in allowed_HGC_names:
                self.coordinate_system = frames.HeliographicCarrington(observer='earth')
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
                
    def _loop_longitudes(self, longitudes: list[float] | np.ndarray[float]):
        """
        Takes discontinuous longitude function and returns a continuous
        longitude function by recursively translating the function at every
        discontinuity due to 0 -> 2pi boundary.

        Parameters
        ----------
        longitudes : list[float] | np.ndarray[float]
            List or np.ndarray of longitudes between 0 and 2pi

        Returns
        -------
        lon : np.ndarray[float]
            np.ndarray of longitudes as a continuous function. To recover
            original longitudes, use lon % 2pi (if lon > pi, lon = lon - 2pi)
        """
        
        longitudes = np.array(longitudes)
        dlon = np.diff(longitudes)
        discontinuities = [abs(dl) > 7*np.pi/4 for dl in dlon]

        if any(discontinuities) == True:
            disc_idx = np.asarray(discontinuities).nonzero()[0]
            lon = longitudes[:disc_idx[0]+1]
            j = 0
            for i, idx in enumerate(disc_idx[:-1]):
                if longitudes[idx] - longitudes[idx+1] < 0:
                    j += 1
                    new_lon = longitudes[idx+1:disc_idx[i+1]+1] - j*2*np.pi
                    lon = np.concatenate((lon, new_lon))
                else:
                    j -= 1
                    new_lon = longitudes[idx+1:disc_idx[i+1]+1] - j*2*np.pi
                    lon = np.concatenate((lon, new_lon))
            j = j+1 if ((longitudes[disc_idx[-1]] - longitudes[disc_idx[-1]+1]) < 0) else j-1
            new_lon = longitudes[disc_idx[-1]+1:] - j*2*np.pi
            lon = np.concatenate((lon, new_lon))
        else:
            lon = longitudes
        
        return lon
                
                
                
    ###########################  PUBLIC FUNCTIONS  ############################
    
    
    def set_coordinates(self, 
                        coordinate_system: frames.SunPyBaseCoordinateFrame = frames.HeliocentricInertial()):
        """
        Query and store coordinates for the times and spacecraft of the 
        Coordinates object.

        Parameters
        ----------
        coordinate_system : sunpy.coordinates.frames.SunPyBaseCoordinateFrame, optional
            Coordinate frame to be used for coordinates representation. 
            The default is frames.HeliocentricInertial().

        Returns
        -------
        None.

        """
        
        self._set_coordinate_system(coordinate_system)

        kernel_keys = {'bepi': 'BEPICOLOMBO MPO', 
                       'earth': 'EARTH', 
                       'psp': 'SOLAR PROBE PLUS',
                       'so': 'SOLAR ORBITER', 
                       'sta': 'STEREO AHEAD'}
        
        for sc in self.spacecraft:
            # Get coordinates for times
            self.sc_coords[sc] = astrospice.generate_coords(
                kernel_keys.get(sc), self.times
                ).transform_to(self.coordinate_system)


    def get_sc_coordinates(self, 
                           spacecraft: str | None = None, 
                           times: np.ndarray[t.Time] | None = None, 
                           coordinate_system: frames.SunPyBaseCoordinateFrame = frames.HeliocentricInertial()):
        """
        Get coordinates from SPICE kernels for a spacecraft for given times in 
        a specified coordinate system. Currently supported spacecraft:
        Solar Orbiter, Parker Solar Probe, BepiColombo, STEREO-A and Earth.
        NOTE: If function is called without keywords, it stores coordinates for
        all self.spacecraft and self.times in self.sc_coords and returns None.

        Parameters
        ----------
        spacecraft : str | None, optional
            Choose one supported spacecraft. If None, coordinates are queried 
            for all self.spacecraft and stored in self.sc_coords. 
            The default is None.
        times : np.ndarray[astropy.time.Time], optional
            Times for which coordinates are queried. If None, 
            coordinates are queried for all self.times and stored in 
            self.sc_coords. The default is None.
        coordinate_system : sunpy.coordinates.frames.SunPyBaseCoordinateFrame, optional
            Coordinate frame coordinates are transformed to. 
            The default is frames.HeliocentricInertial().

        Returns
        -------
        coordinates : np.ndarray[astropy.coordinates.SkyCoord]
            Array of coordinates for chosen spacecraft at given times.*
            *Returns None if called without keywords.

        """
        
        
        # gets coordinates in a specified coordinate system from SPICE kernels for 
        # a set of spacecraft for given times. Currently supported spacecraft:
        # Solar Orbiter, Parker Solar Probe, BepiColombo, STEREO-A and Earth.
        
        self._set_coordinate_system(coordinate_system)

        kernel_keys = {'bepi': 'BEPICOLOMBO MPO', 
                       'earth': 'EARTH', 
                       'psp': 'SOLAR PROBE PLUS',
                       'so': 'SOLAR ORBITER', 
                       'sta': 'STEREO AHEAD'}
        
        if spacecraft == None and times == None:
            for sc in self.spacecraft:
                # Get coordinates for times
                self.sc_coords[sc] = astrospice.generate_coords(
                    kernel_keys.get(sc), self.times
                    ).transform_to(self.coordinate_system)
        else:
            spacecraft = self._parse_sc_names(spacecraft)[0]
            coordinates = astrospice.generate_coords(
                kernel_keys.get(spacecraft), times
                ).transform_to(self.coordinate_system)
            return coordinates
        

    
    # TODO: Update this function
    def get_instantaneous_velocity(self, coords, cartesian_output=True):
        # calculates instantaneous velocity of object at each time based on list 
        # of coordinates corresponding to those times
        
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
#       from a local spk file (unsupported by astrospice)
# TODO: astrospice has been archived - use spiceypy routines instead