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
import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class TimeSeries:
    
    def __init__(self):
        self.variable = str
        self.units = str
        self.times = []
        self.dt = str
        self.bodies = []
        self.coords = []
        self.data = []
        
    def __repr__(self):
        return ('Timeseries for variable {} with units {} and bodies {} between {} and {}'
                .format(self.variable, self.units, self.bodies,
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

    def get_timeseries(self, sim, variable, spacecraft_names=None, times=None, 
                       conj=None, plot=False):

        sim._get_times(mjd=True)
        sim._get_variables()
        
        data = []
        coords = []
        
        omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)
        
        if conj:
            bodies = conj.bodies
            times = conj.times
            
            for i in range(len(conj.times)):
                data_array, units = sim.get_data(variable, conj.times[i])
                values = []
                coord = []
                
                for j in range(len(bodies)):
                    if sim.is_stationary:
                        dt = t.TimeDelta(conj.times[i] - sim.times[0]).to_value('hr')
                        dlon = dt*omega.value
                        phi_idx = sim._closest(sim.phi.value + dlon, 
                                               conj.coords[i][j].lon.rad)
                    else:
                        phi_idx = sim._closest(sim.phi.value, 
                                               conj.coords[i][j].lon.rad)
                    theta_idx = sim._closest(sim.theta.value, 
                                             conj.coords[i][j].lat.rad)
                    r_idx = sim._closest(sim.r.value, 
                                         conj.coords[i][j].distance.au)
                        
                    obstime = sim.times[self._closest(sim.times_mjd, conj.times[i].mjd)]
                    
                    values.append(data_array[phi_idx, theta_idx, r_idx])
                    coord.append(SkyCoord(
                        sim.phi[phi_idx], sim.theta[theta_idx], sim.r[r_idx], 
                        frame = frames.HeliocentricInertial(obstime = obstime))
                        )
                data.append(values)
                coords.append(coord)
            
        else:
            if type(spacecraft_names) == str:
                spacecraft_names = spacecraft_names.split(',')
            
            allowed_names = {'so': ['so', 'solar orbiter', 'solo'], 
                             'psp': ['psp', 'parker solar probe'], 
                             'bepi': ['bepi', 'bepicolombo', 'bepi colombo'], 
                             'sa': ['sa', 'sta', 'stereo-a', 'stereo a', 'stereoa'], 
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
                        'STEREO-A are not yet supported -- got \'{}\''.format(name))
                        
            bodies = sorted(bodies)

            for i in range(len(sim.times)):
    
                data_array, units = sim.get_data(variable, sim.times[i])
                values = []
                coord = []

                for j in range(len(bodies)):
                    if sim.is_stationary:
                        dt = t.TimeDelta(conj.times[i] - sim.times[0]).to_value('hr')
                        dlon = dt*omega.value
                        phi_idx = sim._closest(sim.phi.value + dlon, 
                                               self.coords[j][i].lon.rad)
                    else:
                        phi_idx = sim._closest(sim.phi.value, 
                                               self.coords[j][i].lon.rad)
                    theta_idx = sim._closest(sim.theta.value, 
                                             self.coords[j][i].lat.rad)
                    r_idx = sim._closest(sim.r.value, 
                                         self.coords[j][i].distance.au)
                    values.append(values[phi_idx, theta_idx, r_idx])
                    coord.append(SkyCoord(
                        sim.phi[phi_idx], sim.theta[theta_idx], sim.r[r_idx], 
                        frame = frames.HeliocentricInertial(obstime = times[i]))
                        )
            data.append(values)
            coords.append(coord)
        data = np.asarray(data).transpose()
        coords = np.asarray(coords).transpose()
        
        self.variable = variable
        self.units = units
        self.times = times
        self.dt = sim.dt
        self.coords = coords
        self.bodies = bodies
        self.data = data
        
    def get_truncated_tensor(self, label, new_length=5*u.day, pad_mode='constant', 
                             discard_mostly_padded=True, conj=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # find number of data points per truncated timeseries from chosen 
            # length and timeseries timestep
            num_of_pts = round(new_length.to_value('hr')/self.dt.to_value('hr'))
        except AttributeError:
            if conj:
                num_of_pts = round(new_length.to_value('hr')/conj.dt.to_value('hr'))
            else:
                raise Exception('Timeseries object does not have a timestep '\
                                '\'dt\' defined. Please pass a Conjunction to conj.')
        length = len(self.data[0])
        # find number of truncated timeseries created from each original timeseries
        num_of_subsets = length/num_of_pts
        truncated_data = []
        for data in self.data:
            subsets = []
            # truncate number of subsets to integer - equiv. to int(subsets // 1)
            for s in range(int(num_of_subsets)):
                subsets.append(data[s*num_of_pts:(s+1)*num_of_pts])
            # does the last subset need to be padded? rounded to 2 d.p. due to 
            # precision issue: round(0.999 % 1, 2) will give 1.00
            if round(num_of_subsets % 1, 2) == 0.00 or round(num_of_subsets % 1, 2) == 1.00:
                truncated_ts = subsets
            else:
                last = data[int(num_of_subsets)*num_of_pts:]
                if len(last) < round(num_of_pts/2) and discard_mostly_padded:
                    truncated_ts = subsets
                else:
                    # pad_mode = 'constant' pads with zeros, alternative: pad_mode = 'mean'
                    padded_subset = np.pad(last, (0,num_of_pts-len(last)), mode=pad_mode)
                    subsets.append(padded_subset)
                    truncated_ts = subsets
            if len(truncated_ts) > 0:
                truncated_data.append(truncated_ts)
            #else:
                #print(len(last), round(num_of_pts/2), round(num_of_subsets % 1, 2), truncated_ts)
        if len(truncated_data) > 0:
            truncated_data = np.array(truncated_data)
            truncated_data = torch.tensor(truncated_data, device=device)
            truncated_data = torch.transpose(truncated_data, 0, 1)
            truncated_data = torch.flatten(truncated_data, 1, -1)
            labels = torch.full([len(truncated_data)], float(label), device=device)
        else: # return None if no data
            truncated_data = None
            labels = None
        
        return truncated_data, labels
    
            
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
                        body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
                        colors = ['indianred', 'darkgreen', 'slategrey', 
                                  'steelblue', 'sandybrown', 'slategrey']
                        labels = ['BepiColombo', 'Earth', 'PSP', 
                                  'Solar Orbiter', 'STEREO-A']
                        
                        # FIG 1: cross-correlation plot - max gives lag value
                        title = ('Cross-correlation of timeseries for {} and {}'
                                 .format(labels[body_index.get(self.bodies[i])],
                                         labels[body_index.get(self.bodies[j])]))
                        
                        fig1 = plt.figure(figsize=(8,8), dpi=300)
                        ax = fig1.add_subplot()
                        xc = np.linspace(-len(c)/2, len(c)/2, len(c))
                        ax.plot(xc, c)
                        ax.set_title(title, pad=10, fontsize = 'x-large')
                        
                        # FIG 2: synchronised timeseries - features should overlap
                        title = 'Synchronized timeseries at {}'.format(
                            ', '.join([labels[body_index.get(body)] for body in self.bodies])
                            )
                        
                        fig2 = plt.figure(figsize=(8,8), dpi=300)
                        ax = fig2.add_subplot()
                        ax.set_ylabel('{} [{}]'.format(self.variable, self.units), 
                                      fontsize='large')
                        
                        xs_step_in_hours = t.TimeDelta(xs[1]-xs[0], format='jd').to_value('hr')
                        xs1 = np.arange(0, len(xs))
                        xs2 = xs1.copy() + lag
                        ax.plot(xs1*xs_step_in_hours, spline1(xs), 
                                color=colors[body_index.get(self.bodies[0])], 
                                label=labels[body_index.get(self.bodies[0])])
                        ax.plot(xs2*xs_step_in_hours, spline2(xs), 
                                color=colors[body_index.get(self.bodies[1])], 
                                label=labels[body_index.get(self.bodies[1])])
                        
                        ax.set_title(title, pad=45, fontsize = 'x-large')
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                                  ncol=len(self.bodies), frameon=False, fontsize='large')
                        ax.set_xlabel(r'$\mathrm{Duration \: [hours]}$', 
                                      fontsize='x-large', labelpad=10)

                    lags.append(lag*self.dt.to_value('hr'))
                    pcoefs.append(np.corrcoef(synced_data1, synced_data2))
            ignore_duplicates.append(i)

        return lags, pcoefs
    
        
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
                    phi_idx = self._closest(sim.phi.value + dlon, phi[i])
                else:
                    phi_idx = self._closest(sim.phi.value, phi[i])
                theta_idx = self._closest(sim.theta.value, theta[i])
                r_idx = self._closest(sim.r.value, r[i])
                
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
    