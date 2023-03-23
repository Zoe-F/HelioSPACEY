# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:43:01 2023

@author: Zoe.Faes
"""

# imports
import numpy as np
import astropy.units as u
import astropy.time as t
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.sun.constants import equatorial_radius
import warnings

def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the message
    return '\n' + str(category.__name__) + ': ' + str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

#############################  CONJUNCTION CLASS  ############################

class Conjunction:
    """
    This class stores conjunction parameters for each individual conjunction,
    including category of the conjunction, times, length, bodies involved, 
    coordinates of those bodies, and the timeseries class corresponding to 
    the conjunction.
    """
    def __init__(self, idnum, category, times, dt, bodies, coords):
        self.id = idnum
        self.category = category
        self.times = times
        self.dt = dt
        self.length = t.TimeDelta(times[-1]-times[0])
        self.bodies = bodies
        self.coords = [coords]
        self.isparker = [False]
        self.parkerpairs = []
        self.swspeed = []
        self.timeseries = {}
        
    def __repr__(self):
        return ('Conjunction of the type {} with start time {} and end time {} between {}'
                .format(self.category, self.times[0].iso[0:19], 
                        self.times[-1].iso[0:19], ', '.join(self.bodies)))
    
    def __str__(self):
        return ('Conjunction of the type {} with start time {} and end time {} between {}'
                .format(self.category, self.times[0].iso[0:19],
                        self.times[-1].iso[0:19], ', '.join(self.bodies)))


############################  CONJUNCTIONS CLASS  ############################

class Conjunctions:
    """
    This class finds and stores conjunctions for the specified spacecraft at 
    the specified times, given coordinates obtained from SPICE kernels using 
    the Coordinates class.
    """    
    #############################  CONSTRUCTORS  #############################
    def __init__(self, times, bodies, coords):
        
        print('Initialising Conjunctions...')
        
        self.allconjs = []
        self.nallconjs = int
        
        self.cones = []
        self.quads = []
        self.opps = []
        self.parkers = []
        self.non_conjs = []
        
        self.idnum = int
        
        self.times = times
        self.dt = t.TimeDelta(times[1]-times[0])
        self.bodies = bodies
        self.coords = coords
        
        print('Initialisation complete.')
        
    def __repr__(self):
        return ('There are {} conjunctions between {} from {} to {}'
                .format(self.nallconjs, ', '.join(self.bodies),
                        self.times[0], self.times[-1]))
    
    def __str__(self):
        return ('There are {} conjunctions between {} from {} to {}'
                .format(self.nallconjs, ', '.join(self.bodies),
                        self.times[0], self.times[-1]))
        
    ##########################  PRIVATE FUNCTIONS  ###########################

    # finds the index of the closest element in a list to a given value
    def _closest(self, lst, value):
        lst = np.asarray(lst)
        idx = (np.abs(lst - value)).argmin()
        return idx
    
    # finds angular separation in radians between two points (SkyCoord objects) 
    # using the haversine formula
    def _haversine(self, coords1, coords2):
        dlon = abs(coords1.lon - coords2.lon)
        dlat = abs(coords1.lat - coords2.lat)
        angle = 2 * np.arcsin(np.sqrt(np.sin(dlat/2)**2 + (
            1 - np.sin(dlat/2)**2 - np.sin(coords1.lat/2 + coords2.lat/2)**2
            ) * np.sin(dlon/2)**2))
        return angle
    
    # finds angular separation in radians between two points (SkyCoord objects) 
    # using the Vincenty formula
    def _vincenty(self, coords1, coords2):
        dlon = abs(coords1.lon - coords2.lon)
        angle = np.arctan(np.sqrt(
            (np.cos(coords2.lat) * np.sin(dlon))**2 + 
            (np.cos(coords1.lat) * np.sin(coords2.lat) - 
             np.sin(coords1.lat) * np.cos(coords2.lat) * np.cos(dlon))**2) / 
            (np.sin(coords1.lat) * np.sin(coords2.lat) + 
             np.cos(coords1.lat) * np.cos(coords2.lat) * np.cos(dlon)))
        if angle < 0:
            angle = angle + np.pi*u.rad
        return angle    
    
    # gets angular separation between different spacecraft using their 
    # coordinates and specified formula.
    def _get_sep(self, coords, sep_formula='vincenty'):
        
        seps = []
        ignore_duplicates = []
        
        # Get angular separation
        for i in range(len(self.bodies)):
            for j in range(len(self.bodies)):
                if (i != j) and (j not in ignore_duplicates):
                    sep = []
                    for k in range(len(coords[i])):
                        if sep_formula == 'haversine':
                            sep.append(self._haversine(coords[i][k], coords[j][k]))
                        else:
                            sep.append(self._vincenty(coords[i][k], coords[j][k]))
                    seps.append(sep)
            ignore_duplicates.append(i)
        
        return seps
    
    # finds the longitude along the parker spiral at the requested radius from 
    # the Sun from given coordinates and corresponding solar wind speed.
    # inputs: swspeed, skycoord, and either time increment or radius
    # outputs: skycoord
    def _parker_spiral(self, swspeed, coord, radius):

        day2s = (1*u.day).to(u.s) # seconds in a day
        omega = 2*np.pi*u.rad/(25.38*day2s) # Carrington rate of (sidereal) rotation
        
        r = radius.to(u.au)
        time = coord.obstime + (r - coord.distance)/swspeed # estimate of time for obstime
        # Parker spiral equation
        lon = (coord.lon.to(u.rad) + omega.to(u.rad/u.s) * 
               (coord.distance.to(u.au) - r.to(u.au)) / (swspeed.to(u.au/u.s)))
        
        coordinate = SkyCoord(lon, coord.lat, r, 
                              frame = frames.HeliocentricInertial(obstime = t.Time(time)), 
                              unit = (u.rad, u.rad, u.au))
        
        return coordinate

    # queries the Simulation class for the array of radial solar wind speeds
    # at the coordinates of each spacecraft for all coordinate times
    def _get_swspeed(self, sim):
        
        swspeeds = []
        if sim:
            # need mjd times for easy comparison with coordinate times
            sim._get_times(mjd=True)
        for coord in np.asarray(self.coords).transpose(): 
            if sim:
                # get speeds for closest simulation time
                data, units = sim.get_data('V1', coord[0].obstime)
            swspeed = []
            for c in coord:
                if sim:
                    if sim.is_stationary: # rotate Sun for stationary simulations
                        dt = t.TimeDelta(coord[0].obstime - sim.times[0]).to_value('hr')
                        omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)
                        dlon = dt*omega.value
                        phi_idx = self._closest(sim.phi.value + dlon, c.lon.rad)
                    else:
                        phi_idx = self._closest(sim.phi.value, c.lon.rad)
                    theta_idx = self._closest(sim.theta.value, c.lat.rad)
                    r_idx = self._closest(sim.r.value, c.distance.au)
                    # find and store speed for closest simulation coordinates to S/C location
                    swspeed.append(data[phi_idx, theta_idx, r_idx]*units)
                else:
                    swspeed.append(400*u.km/u.s)
                    warnings.warn('No simulation was given, solar wind speed'\
                                  ' is set to 400 km/s for Parker spiral '\
                                  'conjunctions.')
            swspeeds.append(swspeed)
            
        swspeeds = list(map(list, zip(*swspeeds))) # transpose without converting to np.array
        
        return swspeeds
    
    # finds angular separation at 2.5 Rsun between magnetic field lines 
    # connected to spacecraft using the Parker model and stores the separations
    # for all spacecraft at all coordinate times
    def _get_parker_sep(self, swspeeds):
        
        coords_at_surface = []
        parker_seps = []
        ignore_duplicates = []

        # Find corresponding coordinates at 2.5 Rsun
        for coord, swspeed in zip(self.coords, swspeeds):
            parker_coords = []
            for c, speed in zip(coord, swspeed):
                parker_coord = self._parker_spiral(speed, c, radius=2.5*equatorial_radius)
                parker_coords.append(parker_coord)
            coords_at_surface.append(parker_coords)
            
        # Get angular separation of Parker spiral lines
        for i in range(len(self.bodies)):
            for j in range(len(self.bodies)):
                if (i != j) and (j not in ignore_duplicates):
                    seps = []
                    for k in range(len(coords_at_surface[i])):
                        s = self._vincenty(coords_at_surface[i][k], coords_at_surface[j][k])
                        seps.append(s.to(u.rad))
                    parker_seps.append(seps)
            ignore_duplicates.append(i)
            
        return parker_seps
    
        
    # finds and classifies conjunctions between two spacecraft of the type: 
    # cone, quadrature, opposition or parker spiral.
    # If find_non_conjs == True, creates a Conjunction object for each interval 
    # not corresponing to any conjunction condition and stores it in non_conjs.
    def _get_conjunctions(self, sim, find_non_conjs = True):
        
        cones = []
        quads = []
        opps = []
        parkers = []
        non_conjs = []
        
        # access the correct index for each pair of bodies corresponding to 
        # the separation angle, depending on the number of bodies
        two_body = {'0': [0,1]}
        three_body = {'0': [0,1], '1': [0,2], '2': [1,2]}
        four_body = {'0': [0,1], '1': [0,2], '2': [0,3], 
                     '3': [1,2], '4': [1,3], '5': [2,3]}
        five_body = {'0': [0,1], '1': [0,2], '2': [0,3], '3': [0,4], '4': [1,2], 
                     '5': [1,3], '6': [1,4], '7': [2,3], '8': [2,4], '9': [3,4]}
        number_of_bodies = [two_body, three_body, four_body, five_body]
        
        # get coordinates and separation angles for chosen bodies
        print('Calculating angular separation...')
        seps = self._get_sep(self.coords)
        print('Getting solar wind speed...')
        swspeeds = self._get_swspeed(sim)
        print('Calculating Parker spiral separation...')
        parker_seps = self._get_parker_sep(swspeeds)
        
        # initialise unique id number for each two-body conjunction
        idnum = 0
        
        for s in range(len(seps)):
            print('{}%'.format(s*10)) # progress indicator
            
            # get indices for bodies corresponding to separation angle s
            idx = number_of_bodies[len(self.bodies)-2].get(str(s))
            
            # identify and categorize conjunctions of type cone, quad or opp
            # and store relevant information in Conjunction instance
            for time, sep, parker_sep, coord, speed in zip(
                    self.times, seps[s], parker_seps[s], 
                    np.array(self.coords).transpose(), list(map(list, zip(*swspeeds)))):
                
                non_conj = False
                
                if sep <= 20*u.deg: # cone conjunction
                    conj = Conjunction(
                        idnum=[idnum], 
                        category=['cone'], 
                        times=[time], dt=self.dt, 
                        bodies=np.asarray([self.bodies[idx[0]], self.bodies[idx[1]]]),
                        coords=[coord[idx[0]], coord[idx[1]]]
                        )
                    conj.bodypairs = conj.bodies
                    cones.append(conj)
                    idnum += 1
                    
                elif (sep >= 80*u.deg) and (sep <= 100*u.deg): # quadrature conjunction
                    conj = Conjunction(
                        idnum=[idnum], 
                        category=['quadrature'], 
                        times=[time], dt=self.dt,
                        bodies=np.asarray([self.bodies[idx[0]], self.bodies[idx[1]]]),
                        coords=[coord[idx[0]], coord[idx[1]]]
                        )
                    conj.bodypairs = conj.bodies
                    quads.append(conj)
                    idnum += 1
                    
                elif (sep >= 170*u.deg) and (sep <= 190*u.deg): # opposition conjunction
                    conj = Conjunction(
                        idnum=[idnum], 
                        category=['opposition'], 
                        times=[time], dt=self.dt, 
                        bodies=np.asarray([self.bodies[idx[0]], self.bodies[idx[1]]]),
                        coords=[coord[idx[0]], coord[idx[1]]]
                        )
                    conj.bodypairs = conj.bodies
                    opps.append(conj)
                    idnum += 1
                else:
                    non_conj = True
                
                if parker_sep < 10*u.deg: # Parker conjunction
                    conj = Conjunction(
                        idnum=[idnum], 
                        category=['parker spiral'], 
                        times = [time], dt=self.dt,
                        bodies=np.asarray([self.bodies[idx[0]], self.bodies[idx[1]]]),
                        coords=[coord[idx[0]], coord[idx[1]]])
                    conj.isparker = [True]
                    conj.bodypairs = conj.bodies
                    conj.parkerpairs = conj.bodies
                    conj.swspeed.append([speed[idx[0]], speed[idx[1]]])
                    parkers.append(conj)
                    idnum += 1
                    
                else:
                    if find_non_conjs and non_conj: # store remaining intervals as non-conjs
                        conj = Conjunction(
                            idnum=None, 
                            category=[None], 
                            times = [time], dt=self.dt,
                            bodies=np.asarray([self.bodies[idx[0]], self.bodies[idx[1]]]),
                            coords=[coord[idx[0]], coord[idx[1]]]
                            )
                        non_conjs.append(conj)
                        
        print('100%') # progress indicator complete
        
        # store lists of conjunctions in class attributes cones, quads and opps            
        self.cones = cones
        self.quads = quads
        self.opps = opps
        self.parkers = parkers
        self.non_conjs = non_conjs
        
    # merges conjunction properties
    def _append_conj_properties(self, conj1, conj2):
        
        if conj1.id: # check if conj has id (non_conjs do not have ids)
            conj1.id.append(conj2.id[0])
        conj1.times.append(conj2.times[0])
        conj1.coords.append(conj2.coords[0])
        if conj1.category != conj2.category:
            conj1.category.append(conj2.category)
        if any(conj1.isparker) and any(conj2.isparker):
            # verify that Parker conjunctions are occuring between the same spacecraft
            if all(conj1.parkerpairs) != all(conj2.parkerpairs):
                raise Exception('Check Parker pairs for conjunctions {} and {}'
                                .format(conj1.id, conj2.id))
            else:
                conj1.swspeed.append(conj2.swspeed[0])
        return conj1
        

    # merging conjunctions ensures that conjunctions lasting longer than one
    # time increment are treated as single conjunctions of the correct length
    # input: conjunctions of a single category if separation of cones and Parker 
    # conjunctions is required
    def _merge_consecutive_conjunctions(self, conjunctions):
        
        merged_conjunctions = []
        
        conj = conjunctions[0]
        for j in range(len(conjunctions)-1):
            # merge conjunctions with consecutive times and matching bodies
            if (conjunctions[j].times[-1].isclose(conjunctions[j+1].times[0], self.dt*1.5) 
                and all(conjunctions[j].bodies) == all(conjunctions[j+1].bodies)):
                conj = self._append_conj_properties(conj, conjunctions[j+1])
            
            # separate conjunctions when times are not consecutive or bodies don't match
            elif (not(conjunctions[j].times[-1].isclose(conjunctions[j+1].times[0], self.dt*1.5)) 
                  or conjunctions[j].bodies != conjunctions[j+1].bodies):
                conj.length = t.TimeDelta(conj.times[-1] - conj.times[0])
                merged_conjunctions.append(conj)
                conj = conjunctions[j+1]

            else:
                print('Something funny\'s happening...')
            
            if j == len(conjunctions)-2:
                merged_conjunctions.append(conj)
                        
        merged_conjunctions.sort(key = lambda x: x.times[0])
               
        return merged_conjunctions
    
    # finds conjunctions with at least one body in common occuring at the same 
    # times and stores them as lists of conjunctions
    def _find_simultaneous_conjunctions(self):
        
        conjs = [self.cones, self.quads, self.opps, self.parkers]
        # flatten conjs
        conjunctions = [c for conj in conjs for c in conj]
        conjunctions.sort(key = lambda x: x.times[0])        
        
        # store lists of simultaneous conjunctions in simultaneous
        simultaneous = []
        ignore_duplicates = []
        for i in range(len(conjunctions)):
            for j in range(len(conjunctions)):
                if (i != j) and not (j in ignore_duplicates) and (
                    conjunctions[j].times[-1] >= conjunctions[i].times[0]
                    and conjunctions[j].times[0] <= conjunctions[i].times[-1]):
                    simultaneous.append([conjunctions[i], conjunctions[j]])
            ignore_duplicates.append(i)
            
        return simultaneous
        

    ###########################  PUBLIC FUNCTIONS  ###########################
        
    # finds all conjunctions for bodies and time range specified at class instantiation
    def get_all_conjunctions(self, sim=None, dt=None):
        
        if dt:
            # assign specified timestep
            self.dt = t.TimeDelta(dt)
            # find new times with specified timestep
            self.times = t.Time(
                np.arange(self.times[0], self.times[-1], self.dt)
                )
            
        print('Searching for conjunctions...')
        
        self._get_conjunctions(sim)
        
        print('Merging consecutive conjunctions...')
        
        self.cones = self._merge_consecutive_conjunctions(self.cones) if self.cones else []
        self.quads = self._merge_consecutive_conjunctions(self.quads) if self.quads else []
        self.opps = self._merge_consecutive_conjunctions(self.opps) if self.opps else []
        self.parkers = self._merge_consecutive_conjunctions(self.parkers) if self.parkers else []
        self.non_conjs = self._merge_consecutive_conjunctions(self.non_conjs) if self.non_conjs else []
        
        self.allconjs = [self.cones, self.quads, self.opps, self.parkers]
        self.nallconjs = sum([len(conjs) for cat in self.allconjs for conjs in cat])
        
        print('Consecutive conjunctions merged.')
        
        #print('Identifying multiple spacecraft conjunctions...')
        
        #self.simultaneous = self._find_simultaneous_conjunctions()
        
        #print('Multiple spacecraft conjunctions identified.')
        
        print('Conjunctions search complete.')


    # finds conjunctions according to specified parameters
    def find_conjunctions(self, category=None, spacecraft_names=None,
                          start_time=None, end_time=None, length=None):
        
        out = []
        category_out = []
        start_out = []
        end_out = []
        length_out = []
        bodies_out = []
        
        # parse spacecraft_names
        if spacecraft_names:
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
                    raise Exception(
                        'Invalid spacecraft name. Specify choice with a string'\
                        ' containing the name of a spacecraft, for example: '\
                        '\'Solar Orbiter\' or \'SolO\'. Spacecraft other than'\
                        ' Solar Orbiter, Parker Solar Probe, BepiColombo and '\
                        'STEREO-A are not yet supported -- got \'{}\''
                        .format(name))
            bodies = sorted(bodies)
        else:
            bodies = self.bodies
        
        # flatten conjunctions
        conjunctions = [c for conjs in self.allconjs for c in conjs]
                
        if category:
            # handle string input
            if type(category) == str:
                category = category.split(',')
                category = [cat.strip() for cat in category]
            # if category is non_conj, change set of conjunctions to be searched
            if any(category = [['non_conj'], ['non_conjs'], ['non conj'], ['non conjs'], ['None']]):
                conjunctions = self.non_conjs
                category = [None]
            for conj in conjunctions:
                for i in range(len(conj.category)):
                    for j in range(len(category)):
                        if conj.category[i] == category[j]: # find matching categories
                            category_out.append(conj)
            out.append(category_out)
        if start_time:
            for conj in conjunctions:
                # find matching start times within an hour
                if abs(conj.start.jd - t.Time(start_time).jd) < t.TimeDelta(1*u.hr).to_value('jd'):
                    start_out.append(conj)
            out.append(start_out)
        if end_time:
            for conj in conjunctions:
                # find matching end times within an hour
                if abs(conj.end.jd - t.Time(end_time).jd) < t.TimeDelta(1*u.hr).to_value('jd'):
                    end_out.append(conj)
            out.append(end_out)
        if length:
            for conj in conjunctions:
                # find matching length
                if conj.length == length:
                    length_out.append(conj)
            out.append(length_out)
        if bodies:
            for conj in conjunctions:
                same_bodies = []
                for b in range(len(conj.bodies)):
                    same_bodies.append(conj.bodies[b] == bodies[b])
                if all(same_bodies): # find conjunctions for only the bodies specified
                    bodies_out.append(conj)
            out.append(set(bodies_out))

        # output is the intersection of all the sets for each parameter
        if len(out) == 1:
            out = out[0]
        elif len(out) == 2:
            out = set(out[0]).intersection(set(out[1]))
        elif len(out) == 3:
            out = set(out[0]).intersection(set(out[1]), set(out[2]))
        elif len(out) == 4:
            out = set(out[0]).intersection(set(out[1]), set(out[2]), set(out[3]))
        elif len(out) == 5:
            out = set(out[0]).intersection(set(out[1]), set(out[2]), set(out[3]), set(out[4]))
        
        if len(out) == 0:
            print('No conjunctions found for the specified search parameters.'\
                  ' Try using fewer or different parameters.')
        else:
            print('{} conjunctions correspond to the search parameters.'.format(len(out)))
        
        return list(out)