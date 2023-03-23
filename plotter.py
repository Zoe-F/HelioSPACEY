# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:23:32 2023

@author: Zoe.Faes
"""

#################################  IMPORTS  ##################################

import numpy as np
import astropy.units as u
import astropy.time as t
from sunpy.coordinates import frames
from sunpy.sun import constants as const
import astrospice
import cdflib as cdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.visualization import time_support
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#############################  PRIVATE FUNCTIONS  ############################

 # Function to convert cartesian coordinates to spherical coordinates in radians
 # output: r, lon, lat
def _cartesian2spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    return r, theta, phi

 # Function to convert spherical coordinates in radians to cartesian coordinates
 # input: r, lon, lat
def _spherical2cartesian(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z

#############################  SIMULATION PLOTS  #############################

def plot_ENLIL_slice(sim, variable, time, lat=None, lon=None, radius=None, 
                     cmap='viridis', rotate=None):
    
    plt.rcParams.update({'text.usetex': True, 
                         'font.family': 'Computer Modern Roman'}) # use Tex

    data, units = sim.get_data(variable, time)

    if lat != None and lon == None and radius == None and lat >= min(sim.theta) and lat <= max(sim.theta):
        idx = sim._closest(sim.theta.value, lat.to(u.rad).value)
        data_slice = data[:,idx]
        
        # continuity across phi zero-2pi
        dphi = np.diff(sim.phi).mean()
        wrp_phi = np.concatenate((sim.phi, sim.phi[-1:] + dphi))
        wrp_data_slice = np.concatenate((data_slice, data_slice[0:1, :]), axis=0)
        
        # make figure
        fig = plt.figure(figsize=(8,8), dpi=300)
        # plot options
        ax = fig.add_subplot(projection='polar')
        ax.set_rorigin(0)
        ax.set_rlabel_position(-42.5)
        ax.set_rticks([0.5, 1, 1.5, 2])
        ax.set_title('ENLIL simulation data for solar wind radial velocity '\
                     'in the ecliptic plane', fontsize='x-large', pad=20)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_fontsize('large')
            if cmap == 'viridis':
                label.set_color('white')
            else:
                label.set_color('black')
        tlabels = ax.get_xmajorticklabels()
        for label in tlabels:
            label.set_fontsize('large')
            
        # plot
        if rotate:
            contour = ax.contourf(wrp_phi+rotate, sim.r, wrp_data_slice.transpose(), 
                                  128, cmap = cmap)
        else:
            contour = ax.contourf(wrp_phi, sim.r, wrp_data_slice.transpose(), 
                                  128, cmap = cmap)
        cbar = plt.colorbar(contour, shrink=0.75, pad=0.1)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='{} [{}]'.format(variable, units), fontsize='large', 
                       loc='center')
        
    elif lat == None and lon != None and radius == None:
        idx = sim._closest(sim.phi.value, lon.to(u.rad).value)
        data_slice = data[idx]
        
        # make figure
        fig = plt.figure(figsize=(8,8), dpi=300)
        # plot options
        ax = fig.add_subplot(projection='polar')
        ax.set_rorigin(0)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_rlabel_position(222.5)
        ax.set_rticks([0.5, 1, 1.5, 2])
        ax.set_title('ENLIL simulation data for solar wind radial velocity '\
                     'at {} longitude'.format(round(lon.to(u.deg).value)), 
                     fontsize='x-large', pad=20)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_fontsize('large')
        tlabels = ax.get_xmajorticklabels()
        for label in tlabels:
            label.set_fontsize('large')
            
        # plot
        contour = ax.contourf(sim.theta.value-np.pi/2, sim.r.value, 
                              data_slice.transpose(), 128, cmap = cmap)
        cbar = plt.colorbar(contour, shrink=0.75)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='{} [{}]'.format(variable, units), fontsize='large', 
                       loc='center')
        
        
    elif lat == None and lon == None and radius != None:
        idx = sim._closest(sim.r.value, radius.to(u.au).value)
        data_slice = data[:,:,idx]
        
        # make figure
        fig = plt.figure(figsize=(10,5), dpi=300)
        # plot options
        ax = fig.add_subplot()
        ax.set_xlabel('Longitude [deg]')
        ax.set_xticks([0, np.pi/2, np.pi, np.pi*3/2, np.pi*2], [0,90,180,270,360])
        ax.set_ylabel('Latitude [deg]')
        ax.set_yticks([-np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3], [-60,-30,0,30,60])
        ax.set_title('ENLIL simulation data for solar wind radial velocity '\
                     'at {} AU'.format(round(radius.to(u.au).value)), 
                     fontsize='x-large', pad=20)
        ylabels = ax.get_ymajorticklabels()
        for label in ylabels:
            label.set_fontsize('large')
        xlabels = ax.get_xmajorticklabels()
        for label in xlabels:
            label.set_fontsize('large')
            
        # plot
        contour = ax.contourf(sim.phi.value, sim.theta.value-np.pi/2, 
                              data_slice.transpose(), 128, cmap = cmap)
        cbar = plt.colorbar(contour)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='{} [{}]'.format(variable, units), fontsize='large', 
                       loc='center')
        
    else:
        raise Exception('Please pass a quantity for one of the following: '\
                        'lat, lon or radius. Please note that latitude is only'\
                        ' defined between {} and {} degrees.'.format(
                        np.degrees(min(sim.theta.value)), np.degrees(max(sim.theta.value))
                        ))

    return fig, ax

#############################  TIMESERIES PLOTS  #############################

def plot_timeseries(timeseries):
    
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'})
    #time_support()
    
    # Plotting options
    body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    title = 'Timeseries between %s' % ', '.join([labels[body_index.get(body)] 
                                                 for body in timeseries.bodies])
    
    # Figure set-up
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel('{} [{}]'.format(timeseries.variable, timeseries.units), 
                  fontsize='large')
        
    # Time labelling
    times = []
    time_labels = []
    time_ticks = []
    for i in range(len(timeseries.times)):
        times.append(timeseries.times[i].iso)
        if len(timeseries.times) < 6:
            time_ticks.append(timeseries.times[i].iso)
            time_labels.append(timeseries.times[i].iso[0:10])
        elif len(timeseries.times) > 5 and i%(int(len(timeseries.times)/3)) == 0:
            time_ticks.append(timeseries.times[i].iso)
            time_labels.append(timeseries.times[i].iso[0:10])
    
    # Plot
    for j in range(len(timeseries.bodies)):
        ax.plot(times, timeseries.data[j], color=colors[body_index.get(timeseries.bodies[j])], 
                label=labels[body_index.get(timeseries.bodies[j])])
    ax.set_xticks(time_ticks, labels=time_labels, fontsize='large', usetex=True)
    ax.set_title(title, pad=45, fontsize = 'x-large')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
              ncol=len(timeseries.bodies), frameon=False, fontsize='large')
    ax.set_xlabel(r'$\mathrm{Date}$', fontsize='x-large', labelpad=10)
    #plt.savefig('./Figures/time_series_plot')
    
    return fig


def plot_multiple_timeseries(conjunction, timeseries, show_conjunctions=True, 
                             conjunction_category=['cone', 'parker spiral', 'quadrature', 'opposition']):
                
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'})
    time_support()
    
    # Plotting options
    body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    title = 'Timeseries at %s, %s, %s, %s and %s' %(labels[0], labels[1], 
                                                    labels[2], labels[3], labels[4])
    
    # Figure set-up
    fig = plt.figure(figsize=(9,8), dpi=300)
    ax = []
    
    # Time labelling
    times = []
    time_labels = []
    time_ticks = []
    for i in range(len(timeseries.times)):
        times.append(timeseries.times[i].iso)
        if len(timeseries.times) < 6:
            time_ticks.append(timeseries.times[i].iso)
            time_labels.append(timeseries.times[i].iso[0:10])
        elif len(timeseries.times) > 5 and i%(int(len(timeseries.times)/5)) == 0:
            time_ticks.append(timeseries.times[i].iso)
            time_labels.append(timeseries.times[i].iso[0:10])
    
    # get plot data        
    for i in range(len(timeseries.bodies)):            
        ax.append(fig.add_subplot(len(timeseries.bodies),1,i+1))
        ax[i].plot(times, timeseries.data[i], color=colors[body_index.get(timeseries.bodies[i])], 
                   label=labels[body_index.get(timeseries.bodies[i])])
        
        if show_conjunctions:
            conjunctions = conjunction.find_conjunctions(2, 
                                                         spacecraft_names=[timeseries.bodies[i]], 
                                                         category=conjunction_category)
            for conj in conjunctions:
                if conj.end > timeseries.times[0] or conj.start < timeseries.times[-1]:
                    ax[i].axvspan(conj.start.iso, conj.end.iso, 
                                  color='lightsteelblue', alpha=0.25)
        
        ax[i].set_ylabel(timeseries.variable + ' [' + timeseries.units + ']', 
                         fontsize='large')
        ax[i].set_xlim(times[0], times[-1])
        ax[i].set_xticks(time_ticks, labels=time_labels, fontsize='large', usetex=True)
        ax[i].legend(loc='upper left', bbox_to_anchor=(0, 1.05), handlelength=0 , 
                     handletextpad=0, frameon=False, fontsize='large')
        if i==0:
            ax[i].set_title(title, pad=10, fontsize='x-large')
        if i+1 == len(timeseries.bodies):
            ax[i].set_xlabel(r'$\mathrm{Date}$', fontsize='x-large', labelpad=10)
        else:
            plt.tick_params('x', labelbottom=False)
        #plt.savefig('./Figures/time_series_plot')
    return fig




############################  CONJUNCTIONS PLOTS  ############################

# Plot a given conjunction in 2D
def plot_conjunction2D(coords, conj, idx=0, ENLIL=False, sim=None, variable=None, save_fig=True):
            
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'}) # use TeX
    
    if not float(idx).is_integer():
        raise Exception('index must be an integer.')
    idx = int(idx)
    
    # Plotting options
    body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    
    title = 'Conjunctions between %s in %s coordinates' %(
        ', '.join([labels[body_index.get(body)] for body in conj.bodies]), 
        coords.coordinate_name)
    
    # Figure set-up
    if ENLIL:
        cmap = plt.get_cmap('binary')
        minval=0
        maxval=0.5
        n=128
        my_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), 
            cmap(np.linspace(minval, maxval, n)))
        if sim.is_stationary:
            omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)
            dt = t.TimeDelta(conj.times[idx] - sim.times[0]).to_value('hr')
            angle = dt*omega.value*u.rad
        else:
            angle = None
        fig, ax = plot_ENLIL_slice(sim, variable, time=(conj.times[idx]), 
                                   lat=90*u.deg, cmap=my_cmap, rotate=angle)
    else:
        fig = plt.figure(figsize=(8,8), dpi=300)
        ax = fig.add_subplot(projection='polar')
    
    ax.set_ylim(0,1.1)
    ax.set_rticks([0.25, 0.5, 0.75, 1], fontsize='large')
    ax.set_rlabel_position(-42.5)
    ax.set_title(title, fontsize='x-large', multialignment='center', pad=20)
    
    x = []
    y = []
    
    # Positions of bodies in cartesian coordinates
    if coords.coordinate_system == frames.HeliocentricInertial():
    
        for i in range(len(conj.bodies)):    
            x.append(conj.coords[idx][i].distance.au*np.cos(conj.coords[idx][i].lon.rad))
            y.append(conj.coords[idx][i].distance.au*np.sin(conj.coords[idx][i].lon.rad))
            
        if any(conj.isparker) == True:
            Rsun = const.equatorial_radius.to(u.au)
            day2s = 86400*u.s # seconds in a day
            lon_spiral = []
            counter = 0
            
            for pp in conj.parkerpairs:

                r = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].distance
                lon = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].lon
                swspeed = conj.swspeed[0][np.where(conj.parkerpairs == pp)[0][0]]
                
                r_spiral = np.linspace(0, 1.1, 50)
                lon_surface = lon.to(u.rad) + 2*np.pi*u.rad/(25.38*day2s)*(
                    r.to(u.au) - 2.5*Rsun)/(swspeed.to(u.au/u.s))
                lon_spiral.append(lon_surface.rad + 2*np.pi/(-25.38*day2s.value)*(
                    r_spiral - 2.5*Rsun.value)/(swspeed.to(u.au/u.s).value))
                
                ax.plot(lon_spiral[counter], r_spiral, c=colors[body_index.get(pp)], linestyle='-.')
                counter += 1
          
    if coords.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
        
        for i in range(len(conj.bodies)):    
            x.append(conj.coords[idx][i].radius.au*np.cos(conj.coords[idx][i].lon.rad))
            y.append(conj.coords[idx][i].radius.au*np.sin(conj.coords[idx][i].lon.rad))
        
        if 'parker spiral' in conj.category:
            Rsun = const.equatorial_radius.to(u.au)
            day2s = 86400 # seconds in a day
            lon_spiral = []
            counter = 0
            
            for pp in conj.parkerpairs:
                
                r = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].radius
                lon = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].lon
                
                r_spiral = np.linspace(0, 1.1, 50)
                lon_surface = lon + 2*np.pi*u.rad/(25.38*day2s)*(
                    r.to(u.au) - 2.5*Rsun)/(conj.swspeed*u.km).to(u.au)
                lon_spiral.append(lon_surface.rad + 2*np.pi/(-25.38*day2s)*(
                    r_spiral - 2.5*Rsun.value)/(conj.swspeed*u.km).to(u.au).value)
                
                ax.plot(lon_spiral[counter], r_spiral, c=colors[body_index.get(pp)], linestyle='-.')
                counter += 1
                
          
    lon = []
    for i in range(len(conj.coords[idx])):
        if conj.coords[idx][i].lon.rad < 0:
            lon.append(2*np.pi + conj.coords[idx][i].lon.rad)
        else:
            lon.append(conj.coords[idx][i].lon.rad)
            
    # Parameters for shaded area
    for i in range(0, len(conj.bodypairs), 2):
        theta = np.arange(
            min(lon[np.where(conj.bodies == conj.bodypairs[i])[0][0]], 
                lon[np.where(conj.bodies == conj.bodypairs[i+1])[0][0]]), 
            max(lon[np.where(conj.bodies == conj.bodypairs[i])[0][0]], 
                lon[np.where(conj.bodies == conj.bodypairs[i+1])[0][0]]), 
            0.001)
        line = (-((y[np.where(conj.bodies == conj.bodypairs[i+1])[0][0]] -
            y[np.where(conj.bodies == conj.bodypairs[i])[0][0]]) / (
            x[np.where(conj.bodies == conj.bodypairs[i+1])[0][0]] -
            x[np.where(conj.bodies == conj.bodypairs[i])[0][0]]) * 
            x[np.where(conj.bodies == conj.bodypairs[i])[0][0]]) + 
            y[np.where(conj.bodies == conj.bodypairs[i])[0][0]]) / (
                np.sin(theta) - (
            (y[np.where(conj.bodies == conj.bodypairs[i+1])[0][0]] - 
            y[np.where(conj.bodies == conj.bodypairs[i])[0][0]]) / (
            x[np.where(conj.bodies == conj.bodypairs[i+1])[0][0]] -
            x[np.where(conj.bodies == conj.bodypairs[i])[0][0]]) * 
            np.cos(theta)))
        ax.fill_between(theta, 0, line, facecolor = colors[5], alpha = 0.15)
        
    # Plot!
    if coords.coordinate_system == frames.HeliocentricInertial():
        for i in range(len(conj.bodies)):
            ax.plot([conj.coords[idx][i].lon.rad, conj.coords[idx][i].lon.rad], 
                    [0, conj.coords[idx][i].distance.au], 
                    c=colors[body_index.get(conj.bodies[i])])
                    #, label=labels[body_index.get(conj.bodies[i])])
            ax.scatter(conj.coords[idx][i].lon.rad, conj.coords[idx][i].distance.au, 
                       c=colors[body_index.get(conj.bodies[i])], 
                       label=labels[body_index.get(conj.bodies[i])], s=10)

    if coords.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
        for i in range(len(conj.bodies)):
            ax.plot([conj.coords[idx][i].lon.rad, conj.coords[idx][i].lon.rad], 
                    [0, conj.coords[idx][i].radius.au], 
                    c=colors[body_index.get(conj.bodies[i])], 
                    label=labels[body_index.get(conj.bodies[i])])
    
    if ENLIL:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.25,-0.08), markerscale=2, 
                  labelspacing=0.8, frameon=False, fontsize='medium')
        ax.text(5.23, 1.4, 'Date: {}'.format(str(conj.times[idx].iso)[0:10]), fontsize='medium')
    else: # bbox_to_anchor changed from (-0.08,-0.15)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.08,-0.1), markerscale=1, 
                  labelspacing=0.8, frameon=False, fontsize='medium')
        ax.text(5.3, 1.45, 'Date: {}'.format(str(conj.times[idx].iso)[0:10]), fontsize='medium')
        
        
    tlabels = ax.get_xmajorticklabels()
    for label in tlabels:
        label.set_fontsize('large')
    rlabels = ax.get_ymajorticklabels()
    for label in rlabels:
        label.set_fontsize('large')
    
    #plt.savefig('./Figures/Conjunction_plot.png')
    

# Plot a given conjunction in 3D
def plot_conjunction3D(self, conj, idx=0, save_fig=True, show_grid=True, show_axes=True, 
                       view_elev = 40, view_azim = 0, interactive_fig=False):
            
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'}) # use TeX
    
    if not float(idx).is_integer():
        raise Exception('Index must be an integer.')
    idx = int(idx)
    
    sat_coords = []
    
    # Get coordinates for times
    sat_coords.append(astrospice.generate_coords('BEPICOLOMBO MPO', self.times))
    sat_coords.append(astrospice.generate_coords('EARTH', self.times))
    sat_coords.append(astrospice.generate_coords('SOLAR PROBE PLUS', self.times))
    sat_coords.append(astrospice.generate_coords('SOLAR ORBITER', self.times))
    sat_coords.append(astrospice.generate_coords('STEREO AHEAD', self.times))

    # Coordinate transform
    for i in range(len(sat_coords)):
        sat_coords[i] = sat_coords[i].transform_to(self.coordinate_system)
    
    # Plotting options
    body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    
    date = str((conj.start + idx*self.dt).iso)[0:10]
    
    title = 'Conjunctions between %s in %s coordinates' %(
        ', '.join([labels[body_index.get(body)] for body in conj.bodies]), 
        self.coordinate_name)
    
    # Figure set-up
    if interactive_fig:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        scalebar = AnchoredSizeBar(ax.transData, 0.05, '1 AU', loc = 'lower center', 
                                   pad=0, sep=5, color='black', frameon=False)
        titlepad = 10
    else:
        fig = plt.figure(figsize=(8,8), dpi=300)
        ax = fig.add_subplot(projection='3d')
        scalebar = AnchoredSizeBar(ax.transData, 0.05, '1 AU', loc = 'lower center', 
                                   pad=7, sep=5, color='black', frameon=False)
        titlepad = 10
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_zlim(-0.5,0.5)
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.grid(show_grid)
    if not show_axes:
        ax.add_artist(scalebar)
        ax.set_axis_off()
    ax.set_title(title, fontsize='large', multialignment='center', pad = titlepad)
    
    # Positions of bodies in cartesian coordinates
    if self.coordinate_system == frames.HeliocentricInertial():
    
        for i in range(len(conj.bodies)):
            # Plot spacecraft paths
            xsat, ysat, zsat = self._spherical2cartesian(
                sat_coords[body_index.get(conj.bodies[i])].distance.au, 
                np.pi + sat_coords[body_index.get(conj.bodies[i])].lon.rad, 
                np.pi/2 + sat_coords[body_index.get(conj.bodies[i])].lat.rad)
            ax.scatter(xsat, ysat, zsat, c=colors[body_index.get(conj.bodies[i])], s=0.2)
            # Plot conjunctions
            x, y, z = self._spherical2cartesian(
                conj.coords[idx][i].distance.au, np.pi + conj.coords[idx][i].lon.rad, 
                np.pi/2 + conj.coords[idx][i].lat.rad)
            ax.plot([0, x], [0, y], [0, z], c=colors[body_index.get(conj.bodies[i])], 
                    label=labels[body_index.get(conj.bodies[i])])
            
        # Parameters for shaded area
        for i in range(0, len(conj.bodypairs), 2):
            coords0 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[i])[0][0]]
            coords1 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[i+1])[0][0]]
            x0, y0, z0 = self._spherical2cartesian(
                coords0.distance.au, np.pi + coords0.lon.rad, np.pi/2 + coords0.lat.rad)
            x1, y1, z1 = self._spherical2cartesian(
                coords1.distance.au, np.pi + coords1.lon.rad, np.pi/2 + coords1.lat.rad)
            verts = [np.array([0,0,0]), np.array([x0,y0,z0]), np.array([x1,y1,z1])]
            ax.add_collection3d(Poly3DCollection([verts], facecolor = colors[5], alpha = 0.2))
    
    # Plot!
    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
        
        for i in range(len(conj.bodies)):
            # Plot spacecraft paths
            xsat, ysat, zsat = self._spherical2cartesian(
                sat_coords[body_index.get(conj.bodies[i])].radius.au, 
                np.pi + sat_coords[body_index.get(conj.bodies[i])].lon.rad, 
                np.pi/2 + sat_coords[body_index.get(conj.bodies[i])].lat.rad)
            ax.scatter(xsat, ysat, zsat, c=colors[body_index.get(conj.bodies[i])], s=0.2)
            # Plot conjunctions
            x, y, z = self._spherical2cartesian(
                conj.coords[idx][i].radius.au, np.pi + conj.coords[idx][i].lon.rad, 
                np.pi/2 + conj.coords[idx][i].lat.rad)
            ax.plot([0, x], [0, y], [0, z], c=colors[body_index.get(conj.bodies[i])], 
                    label=labels[body_index.get(conj.bodies[i])])
            
        # Parameters for shaded area
        for i in range(0, len(conj.bodypairs), 2):
            coords0 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[i])[0][0]]
            coords1 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[i+1])[0][0]]
            x0, y0, z0 = self._spherical2cartesian(
                coords0.radius.au, np.pi + coords0.lon.rad, np.pi/2 + coords0.lat.rad)
            x1, y1, z1 = self._spherical2cartesian(
                coords1.radius.au, np.pi + coords1.lon.rad, np.pi/2 + coords1.lat.rad)
            verts = [np.array([0,0,0]), np.array([x0,y0,z0]), np.array([x1,y1,z1])]
            ax.add_collection3d(Poly3DCollection([verts], facecolor = colors[5], alpha = 0.2))
           
    ax.legend(loc='best', markerscale=2, labelspacing=0.8, edgecolor='white', framealpha=1)
    ax.text(-1,-1.4,0.55, 'Date: %s' %(str((conj.start + idx*self.dt).iso)[0:10]), fontsize='large') 
    #(1,1.4,0.5) for azim=180
    ax.view_init(elev=view_elev, azim=view_azim)
    
    #fig.savefig('./Figures/3D Conjunctions for poster %s.png' %conj.start.iso[0:10])
        

##########################  CONJUNCTIONS ANIMATIONS  #########################


def animate_conjunctions2D(self, ENLIL=False, variable=None, sim_file_paths=None):
    
    # TODO: disable artifical sun rotation when using time-varying cdf files
    
    # Get coordinates for times
    so_coords = astrospice.generate_coords('SOLAR ORBITER', self.times)
    psp_coords = astrospice.generate_coords('SOLAR PROBE PLUS', self.times)
    sta_coords = astrospice.generate_coords('STEREO AHEAD', self.times)
    bepi_coords = astrospice.generate_coords('BEPICOLOMBO MPO', self.times)
    earth_coords = astrospice.generate_coords('EARTH', self.times)

    # Coordinate transform
    so_coords = so_coords.transform_to(self.coordinate_system)
    psp_coords = psp_coords.transform_to(self.coordinate_system)
    sta_coords = sta_coords.transform_to(self.coordinate_system)
    bepi_coords = bepi_coords.transform_to(self.coordinate_system)
    earth_coords = earth_coords.transform_to(self.coordinate_system)
    
    # Plotting options
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'}) # use TeX
    body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'Stereo-A']
    title = 'Conjunctions between SolO, PSP, BepiColombo, Stereo-A and Earth'\
        ' in %s coordinates' %(self.coordinate_name)
    
    if ENLIL:
        cmap = plt.get_cmap('binary')
        minval=0
        maxval=0.33
        n=128
        my_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), 
            cmap(np.linspace(minval, maxval, n)))
        
        sim_times = self._get_sim_times(sim_file_paths)
        for j in range(len(sim_times)):
            sim_times[j] = sim_times[j].mjd
        
    else:
        fig = plt.figure(figsize=(8.5,8), dpi=300)
        ax = fig.add_subplot(projection='polar')
    
    omega = 2*np.pi*u.rad/(25.38*u.day)
        
    def animate(i):
        
        ax.clear()
        
        ax.set_ylim(0,1.1)
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_rlabel_position(-42.5)
        ax.set_title(title,fontsize='x-large', pad=20)
        
        count = 'Date: %s' %str((self.starttime + self.dt*i).iso)[0:10]
        ax.text(4,1.6, 'Date: 0000-00-00', color='w', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
        ax.text(4,1.6, count)
                    
        conjs = {2: self.twobodyconjs, 3: self.threebodyconjs, 4: self.fourbodyconjs, 5: self.fivebodyconjs}
        
        if ENLIL:
            #TODO: add netcdf support
            idx = self._closest(sim_times, (self.starttime + self.dt*i).mjd)
            file = cdf.CDF(sim_file_paths[idx])
            
            # get data
            var = file.varget(variable=variable)[0]
            var = np.reshape(var, (180,60,320)) # lon, lat, r ie. phi, theta, r
            #units = file.varattsget(variable=variable, expand=True).get('units')[0]
            
            # get r, phi, theta coordinates data are defined on
            r = file.varget(variable='r')[0]
            r = (r*u.m).to(u.au).value
            phi = file.varget(variable='phi')[0]
            theta = file.varget(variable='theta')[0]
            
            ax.set_rorigin(0)
            # continuity across phi zero-2pi
            var_slice = var[:,self._closest(theta, np.pi/2)]
            dphi = np.diff(phi).mean()
            wrp_phi = np.concatenate((phi, phi[-1:] + dphi))
            wrp_var_slice = np.concatenate((var_slice, var_slice[0:1, :]), axis=0)            
                
            # plot
            contour = ax.contourf(wrp_phi+i*omega.value, r, wrp_var_slice.transpose(), 128, cmap = my_cmap)
            
        # Plot spacecraft paths
        if self.coordinate_system == frames.HeliocentricInertial():
            so = ax.scatter(so_coords.lon.to(u.rad)[i], so_coords.distance.to(u.au)[i], c=colors[3], s=5)
            psp = ax.scatter(psp_coords.lon.to(u.rad)[i], psp_coords.distance.to(u.au)[i], c=colors[2], s=5)
            bepi = ax.scatter(bepi_coords.lon.to(u.rad)[i], bepi_coords.distance.to(u.au)[i], c=colors[0], s=5)
            sta = ax.scatter(sta_coords.lon.to(u.rad)[i], sta_coords.distance.to(u.au)[i], c=colors[4], s=5)
            earth = ax.scatter(earth_coords.lon.to(u.rad)[i], earth_coords.distance.to(u.au)[i], c=colors[1], s=5)
        if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
            so = ax.scatter(so_coords.lon.to(u.rad)[i], so_coords.radius.to(u.au)[i], c=colors[3], s=5)
            psp = ax.scatter(psp_coords.lon.to(u.rad)[i], psp_coords.radius.to(u.au)[i], c=colors[2], s=5)
            bepi = ax.scatter(bepi_coords.lon.to(u.rad)[i], bepi_coords.radius.to(u.au)[i], c=colors[0], s=5)
            sta = ax.scatter(sta_coords.lon.to(u.rad)[i], sta_coords.radius.to(u.au)[i], c=colors[4], s=5)
            earth = ax.scatter(earth_coords.lon.to(u.rad)[i], earth_coords.radius.to(u.au)[i], c=colors[1], s=5)

        ax.legend([so, psp, bepi, earth, sta], [labels[3], labels[2], labels[0], labels[1], labels[4]], 
                  loc='lower right', bbox_to_anchor=(1.1,-0.1), markerscale=1.5, 
                  labelspacing=0.8, frameon=False)
        
            
        # Plot conjunctions
        for key in conjs.keys():
            for conj in conjs.get(key):
                if conj.start == (self.starttime + self.dt*i):
                    idx=0
                    x = []
                    y = []
                    
                    # different attribute names between HCI and HGC coordinates
                    if self.coordinate_system == frames.HeliocentricInertial():
                        # Positions of bodies in cartesian coordinates
                        for j in range(key):
                            x.append(conj.coords[idx][j].distance.au*np.cos(conj.coords[idx][j].lon.rad))
                            y.append(conj.coords[idx][j].distance.au*np.sin(conj.coords[idx][j].lon.rad))
                            
                        if any(conj.isparker) == True:
                            Rsun = const.equatorial_radius.to(u.au)
                            day2s = 86400*u.s # seconds in a day
                            lon_spiral = []
                            counter = 0
                            
                            for pp in conj.parkerpairs:

                                r = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].distance
                                lon = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].lon
                                #print(conj.swspeed, key, conj, self.starttime+self.dt*i)
                                #, type(conj.swspeed[0][0]), type(conj.swspeed[0][0][0]))
                                swspeed = conj.swspeed[0][np.where(conj.parkerpairs == pp)[0][0]]
                                
                                r_spiral = np.linspace(0, 1.1, 50)
                                lon_surface = lon.to(u.rad) + 2*np.pi*u.rad/(25.38*day2s)*(
                                    r.to(u.au) - 2.5*Rsun)/(swspeed.to(u.au/u.s))
                                lon_spiral.append(
                                    lon_surface.rad + 2*np.pi/(-25.38*day2s.value)*(
                                        r_spiral - 2.5*Rsun.value)/(swspeed.to(u.au/u.s).value))
                                
                                ax.plot(lon_spiral[counter], r_spiral, 
                                        c=colors[body_index.get(pp)], linestyle='-.')
                                counter += 1
                                
                    # different attribute names between HCI and HGC coordinates
                    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
                            # Positions of bodies in cartesian coordinates
                            for j in range(key):
                                x.append(conj.coords[idx][j].radius.au*np.cos(conj.coords[idx][j].lon.rad))
                                y.append(conj.coords[idx][j].radius.au*np.sin(conj.coords[idx][j].lon.rad))
                                
                            if 'parker spiral' in conj.category:
                                Rsun = const.equatorial_radius.to(u.au)
                                day2s = 86400 # seconds in a day
                                lon_spiral = []
                                counter = 0
                                
                                for pp in conj.parkerpairs:

                                    r = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].radius
                                    lon = conj.coords[idx][np.where(conj.bodies == pp)[0][0]].lon
                                    
                                    r_spiral = np.linspace(0, 1.1, 50)
                                    lon_surface = lon + 2*np.pi*u.rad/(25.38*day2s.value)*(
                                        r.to(u.au) - 2.5*Rsun)/(conj.swspeed).to(u.au/u.s).value
                                    lon_spiral.append(
                                        lon_surface.rad + 2*np.pi/(-25.38*day2s)*
                                        (r_spiral - 2.5*Rsun.value)/
                                        (conj.swspeed*u.km).to(u.au).value)
                                    
                                    ax.plot(lon_spiral[counter], r_spiral, 
                                            c=colors[body_index.get(pp)], linestyle='-.')
                                    counter += 1
                                    
                    for j in range(0, len(conj.bodypairs), 2):
                        coords0 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[j])[0][0]]
                        coords1 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[j+1])[0][0]]
                        verts = [np.array([0,0]), 
                                 np.array([coords0.lon.rad,coords0.distance.au]), 
                                 np.array([coords1.lon.rad,coords1.distance.au])]
                        ax.add_collection(
                            mpl.collections.PolyCollection([verts], facecolor = colors[5], alpha = 0.2))
                    
                    # Plot!
                    if self.coordinate_system == frames.HeliocentricInertial():
                        for j in range(len(conj.bodies)):
                            ax.plot([conj.coords[idx][j].lon.rad, conj.coords[idx][j].lon.rad], 
                                    [0, conj.coords[idx][j].distance.au], 
                                    c=colors[body_index.get(conj.bodies[j])], 
                                    label=labels[body_index.get(conj.bodies[j])])

                    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
                        for j in range(len(conj.bodies)):
                            ax.plot([conj.coords[idx][j].lon.rad, conj.coords[idx][j].lon.rad], 
                                    [0, conj.coords[idx][j].radius.au], 
                                    c=colors[body_index.get(conj.bodies[j])], 
                                    label=labels[body_index.get(conj.bodies[j])])
        
        # if ENLIL and i==0:
        #     cbar = plt.colorbar(contour, shrink=0.75, pad=0.1)
        #     cbar.ax.tick_params(labelsize=12)
        #     cbar.set_label(label=r'$\mathrm{%s \; [%s]}$' %(variable, units), fontsize='large', loc='center')
                
        if i%20 == 0:
            print(str(int(i/len(self.times)*100)) + '% complete.')
        
        plt.show()
        
    orbits = FuncAnimation(fig, animate, frames=np.arange(len(self.times)), interval=20)
    print('Saving animation to .mp4 file. This may take several minutes.')
    orbits.save('./Animations/Conjunctions %s to %s.mp4' %(
        str(self.starttime)[0:10], str(self.endtime)[0:10]), 
        writer = 'ffmpeg', fps = 10)
    print('Save completed.')
    #print('Generating html5 video')
    #link = orbits.to_html5_video()
    #print(link)
    

def animate_conjunctions3D(self):
    
    sat_coords = []
    
    # Get coordinates for times
    sat_coords.append(astrospice.generate_coords('BEPICOLOMBO MPO', self.times))
    sat_coords.append(astrospice.generate_coords('EARTH', self.times))
    sat_coords.append(astrospice.generate_coords('SOLAR PROBE PLUS', self.times))
    sat_coords.append(astrospice.generate_coords('SOLAR ORBITER', self.times))
    sat_coords.append(astrospice.generate_coords('STEREO AHEAD', self.times))

    # Coordinate transform
    for i in range(len(sat_coords)):
        sat_coords[i] = sat_coords[i].transform_to(self.coordinate_system)
    
    # Plotting options
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'}) # use TeX
    body_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = [None, 'BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'Stereo-A']
    title = 'Conjunctions between SolO, PSP, BepiColombo, Stereo-A and Earth'\
        ' in %s coordinates' %(self.coordinate_name)
    
    fig = plt.figure(figsize=(9,8), dpi=300)
    ax = fig.add_subplot(projection='3d')
    
    view_elev = np.linspace(60, 15, len(self.times))
    view_azim = 90 #np.linspace(-150, 210, len(self.times))
    
    def animate(i):
        
        ax.clear()
        
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.set_zlim(-0.5,0.5)
        ax.set_xlabel('X [AU]')
        ax.set_ylabel('Y [AU]', labelpad=15)
        ax.set_zlabel('Z [AU]', labelpad=15)
        ax.tick_params(axis='y', which='major', pad=10)
        ax.tick_params(axis='z', which='major', pad=10)
        #scalebar = AnchoredSizeBar(ax.transData, 0.05, '1 AU', loc = 'lower center', 
        #                           pad=0, sep=5, color='black', frameon=False)
        ax.grid(False)
        #ax.add_artist(scalebar)
        #ax.set_axis_off()
        ax.set_title(title, fontsize='large', multialignment='center')
        ax.view_init(elev=view_elev[i], azim=view_azim)
        
        # -2, -1.5, 0
        date = 'Date: %s \n \n' %str((self.starttime + self.dt*i).iso)[0:10]
        labels[0] = date
        ax.scatter(0,0,0, c='w', alpha=0)
        #ax.text(-70,1,0, 'Date: 0000-00-00', transform = fig.transFigure, 
        #        color='w', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
        #ax.text(-70,1,0, date, transform = fig.transFigure)
        
        # Plot spacecraft paths
        # different attribute names between HCI and HGC coordinates
        if self.coordinate_system == frames.HeliocentricInertial():
            for j in range(len(self.bodies)):
                xsat = []; ysat = []; zsat = []
                for k in range(i):
                    x, y, z = self._spherical2cartesian(
                        sat_coords[j][k].distance.au, np.pi + sat_coords[j][k].lon.rad, 
                        np.pi/2 + sat_coords[j][k].lat.rad)
                    xsat.append(x)
                    ysat.append(y)
                    zsat.append(z)
                ax.scatter(xsat, ysat, zsat, c=colors[j], s=2)
        # different attribute names between HCI and HGC coordinates
        if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
            for j in range(len(self.bodies)):
                xsat = []; ysat = []; zsat = []
                for k in range(i):
                    x, y, z = self._spherical2cartesian(
                        sat_coords[j][k].radius.au, np.pi + sat_coords[j][k].lon.rad, 
                        np.pi/2 + sat_coords[j][k].lat.rad)
                    xsat.append(x)
                    ysat.append(y)
                    zsat.append(z)
                ax.scatter(xsat, ysat, zsat, c=colors[j], s=2)
                
        ax.legend(labels = labels, loc='lower right', bbox_to_anchor=(1.15,0.4), 
                  markerscale=1.5, handletextpad=0.5, edgecolor='white', 
                  framealpha=1)#frameon=False)
        
        conjs = {2: self.twobodyconjs, 3: self.threebodyconjs, 4: self.fourbodyconjs, 5: self.fivebodyconjs}
        
        # Plot conjunctions
        for key in conjs.keys():
            for conj in conjs.get(key):
                if conj.start == (self.starttime + self.dt*i):
                    idx=0
                    x = []
                    y = []
                    z = []
                    
                    # different attribute names between HCI and HGC coordinates
                    if self.coordinate_system == frames.HeliocentricInertial():
                        # Positions of bodies in cartesian coordinates
                        for j in range(key):
                            x, y, z = self._spherical2cartesian(
                                conj.coords[idx][j].distance.au, 
                                np.pi + conj.coords[idx][j].lon.rad, 
                                np.pi/2 + conj.coords[idx][j].lat.rad)

                    # Parameters for shaded area
                    for j in range(0, len(conj.bodypairs), 2):
                        coords0 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[j])[0][0]]
                        coords1 = conj.coords[idx][np.where(conj.bodies == conj.bodypairs[j+1])[0][0]]
                        x0, y0, z0 = self._spherical2cartesian(
                            coords0.distance.au, np.pi + coords0.lon.rad, np.pi/2 + coords0.lat.rad)
                        x1, y1, z1 = self._spherical2cartesian(
                            coords1.distance.au, np.pi + coords1.lon.rad, np.pi/2 + coords1.lat.rad)
                        verts = [np.array([0,0,0]), np.array([x0,y0,z0]), np.array([x1,y1,z1])]
                        ax.add_collection3d(Poly3DCollection([verts], facecolor = colors[5], alpha = 0.2))
                     
                        
                    # Plot!
                    if self.coordinate_system == frames.HeliocentricInertial():
                        for j in range(len(conj.bodies)):
                            xsat, ysat, zsat = self._spherical2cartesian(
                                conj.coords[idx][j].distance.au, 
                                np.pi + conj.coords[idx][j].lon.rad, 
                                np.pi/2 + conj.coords[idx][j].lat.rad)
                            ax.plot([0, xsat], [0, ysat], [0, zsat], 
                                    c=colors[body_index.get(conj.bodies[j])], 
                                    label=labels[body_index.get(conj.bodies[j])])

                    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
                        for j in range(len(conj.bodies)):
                            xsat, ysat, zsat = self._spherical2cartesian(
                                conj.coords[idx][j].radius.au, 
                                np.pi + conj.coords[idx][j].lon.rad, 
                                np.pi/2 + conj.coords[idx][j].lat.rad)
                            ax.plot([0, xsat], [0, ysat], [0, zsat], 
                                    c=colors[body_index.get(conj.bodies[j])], 
                                    label=labels[body_index.get(conj.bodies[j])])
                
        if i%20 == 0:
            print(str(int(i/len(self.times)*100)) + '% complete.')
        
        plt.show()
        
    orbits = FuncAnimation(fig, animate, frames=np.arange(len(self.times)), interval=30)
    print('Saving animation to .mp4 file. This may take several minutes.')
    orbits.save('./Animations/Conjunctions 3D view 3 %s to %s.mp4' %(
        str(self.starttime)[0:10], str(self.endtime)[0:10]), 
        writer = 'ffmpeg', fps = 10)
    print('Save completed.')
    #print('Generating html5 video')
    #link = orbits.to_html5_video()
    #print(link)