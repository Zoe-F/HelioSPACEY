# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:23:32 2023

@author: Zoe.Faes
"""

# TODO: implement function animate(plotter.function) to animate relevant plots produced with heliospacey

#################################  IMPORTS  ##################################

import numpy as np
import astropy.units as u
import astropy.time as t
from simulation import Simulation
from coordinates import Coordinates
from sunpy.coordinates import frames
from sunpy.sun import constants as const
import astrospice
import cdflib as cdf
from scipy.interpolate import CubicSpline, interp1d
from sklearn.preprocessing import minmax_scale
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.visualization import time_support
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from matplotlib.collections import LineCollection
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

def _symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))

def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def _make_logscale(min_val, max_val):
    if np.sign(min_val) != np.sign(max_val):
        neg_levels = np.geomspace(abs(min_val), 1e-12, 50)
        pos_levels = np.geomspace(1e-12, max_val, 50)
        levels = np.concatenate([-1*neg_levels, pos_levels])
    elif np.sign(max_val) == -1:
        levels = np.geomspace(abs(min_val), abs(max_val), 100)
        levels = -1 * levels
    else:
        levels = np.geomspace(min_val, max_val, 100)
    if levels[0] - levels[-1] > 0:
        levels = levels[::-1]
    norm = mpl.colors.SymLogNorm(1e-12)
    
    return levels, norm

# my_cmap = truncate_colormap(mpl.colormaps['viridis'], minval=0.4, maxval=1.0)


#############################  SIMULATION PLOTS  #############################

def plot_ENLIL_slice(sim, variable, time, lat=None, lon=None, radius=None, 
                     cmap='viridis', logscale=False, rotate=None):
    
    data, units = sim.get_data(variable, time)

    if lat != None and lon == None and radius == None and lat >= min(sim.theta) and lat <= max(sim.theta):
        idx = sim._nearest(sim.theta.value, lat.to(u.rad).value)
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
            
        if logscale:
            levels, norm = _make_logscale(data_slice.min(), data_slice.max())
        else:
            levels = 128
            norm = None
            
        # plot
        if rotate != None:
            contour = ax.contourf(wrp_phi+rotate, sim.r, wrp_data_slice.transpose(), 
                                  levels = levels, cmap = cmap, norm = norm)
        else:
            contour = ax.contourf(wrp_phi, sim.r, wrp_data_slice.transpose(), 
                                  levels = levels, cmap = cmap, norm=norm)
        cbar = plt.colorbar(contour, shrink=0.75, pad=0.1)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='{} [{}]'.format(variable, units), fontsize='large', 
                       loc='center')
        
    elif lat == None and lon != None and radius == None:
        idx = sim._nearest(sim.phi.value, lon.to(u.rad).value)
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
        idx = sim._nearest(sim.r.value, radius.to(u.au).value)
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


def plot_vector_field(sim: Simulation | str, 
                      vector: str,
                      color_parameter: str = 'magnitude', 
                      cmap: mpl.colors.Colormap | str = 'viridis', 
                      background_var: str | None = None,
                      flow_paths: list | np.ndarray | None = None,
                      flow_rate_markers: list | None = None,
                      time: t.Time | str | None = None):
    """
    Plot a vector field (velocity or magnetic field) from simulation data.
    Options to plot plasma flow paths and a contour plot of another parameter
    in the background.

    Parameters
    ----------
    sim : heliospacey.Simulation object | str
        The simulation object containing the required vector field information,
        or the path to the simulation file.
    vector : str
        Choose from velocity 'V' or magnetic field 'B'.
    color_parameter : str, optional
        The quantity used to determine the colors of the arrows. Choose from
        'magnitude' to highlight the change in magnitude in the field, 
        'tangent' to highlight any deviation from a purely radial field, 
        or 'third component' to reflect the latitudinal component of the field. 
        The default is 'magnitude'.
    cmap : matplotlib.colors.Colormap | str, optional
        Color map used for plotting. The default is 'viridis'.
    background_var : str | None, optional
        Specify a variable to plot a greyscale contour under the vector field, 
        for example: 'B3', 'V1', 'D' or 'T'. The default is None.
    flow_paths: list | numpy.array | None, optional
        List or array of 3D spherical coordinates to plot on the vector field.
    time : astropy.time.Time | str | None, optional
        Specify time if simulation is time-dependent. The default is None.

    Returns
    -------
    'matplotlib.pyplot.Figure'

    """
    
    ###########################   PARSE INPUTS   ##############################
    
    # instantiate sim if file path given
    if isinstance(sim, str):
        sim = Simulation(sim)
    
    # parse vector
    allowed_vector = {'V': ['v', 'vel', 'velocity'], 
                      'B': ['b', 'mag', 'magnetic field', 'b field', 'magnetic']}
    if vector.lower() in allowed_vector['V']:
        vector = 'V'
        var_r = 'V1'
        var_phi = 'V3'
    elif vector.lower() in allowed_vector['B']:
        vector = 'B'
        var_r = 'B1'
        var_phi = 'B3'
    else:
        raise Exception("Invalid argument for vector. Please choose either "\
                        "'V' for velocity, or 'B' for magnetic field.")
    
    # parse color_parameter
    allowed_color_parameter = ['magnitude', 'mag', 'tangent', 'tan', 'third component', 'third', '3rd', '3D', 'latitudinal', 'lat']
    if color_parameter.lower() in allowed_color_parameter[0:2]:
        color_parameter = 'Magnitude |v|' if vector == 'V' else 'Magnitude |B|'
    elif color_parameter.lower() in allowed_color_parameter[2:4]:
        color_parameter = 'Tangent Ratio ($v_{lon}/v_r$)' if vector == 'V' else 'Tangent Ratio ($B_{lon}/B_r$)'
    elif color_parameter.lower() in allowed_color_parameter[4:]:
        color_parameter = 'Latitudinal Velocity' if vector == 'V' else 'Latitudinal Magnetic Field'
    else:
        raise Exception("Invalid argument for color_parameter. Please choose "\
                        "from 'magnitude', 'tangent' or 'third component'.")
    
    # parse background_var
    if background_var:
        not_found = True
        for key, values in sim.allowed_var_names.items():
            if background_var.lower() in values:
                background_var = key
                not_found = False
                break
        if not_found:
            raise Exception("Invalid argument for background_var. Please "\
                            "choose from {}".format(list(sim.variables.keys())[3:]))
    
    ###########################   PROCESS DATA   ##############################
    
    # specify latitude for data slice
    lat = 90*u.deg

    # get vector field
    data, units = sim.get_data(var_r, time=time)
    idx = sim._nearest(sim.theta.value, lat.to(u.rad).value)
    vr = data[:,idx]
    data, units = sim.get_data(var_phi, time=time)
    idx = sim._nearest(sim.theta.value, lat.to(u.rad).value)
    vphi = data[:,idx]
    
    # TODO: check magnitude for magnetic field (should be on the order of 1e-08 Teslas)
    # TODO: length of arrows for magnetic field should decrease with distance from sun
    
    if vector == 'B':
        vr = _symlog(vr)
        vphi = _symlog(vphi)
    
    # transform components to cartesian for plt.quiver input          
    phi, r = np.meshgrid(sim.phi.value, sim.r.value)
    vx = np.transpose(vr) * np.cos(phi) - np.transpose(vphi) * np.sin(phi)
    vy = np.transpose(vr) * np.sin(phi) + np.transpose(vphi) * np.cos(phi)
    
    # mask values for a legible plot (vector field is otherwise too densely populated)
    mask = np.ones_like(vx, dtype=bool)
    mask[::35, ::4] = False  # Set every (nth, mth) element to True
    vx_masked = np.ma.masked_array(vx, mask)
    vy_masked = np.ma.masked_array(vy, mask)
    
    if color_parameter[:3].lower() == 'mag':
        color_values = np.hypot(vx_masked, vy_masked)
    elif color_parameter[:3].lower() == 'tan':
        color_values = np.arctan2(vr, vphi)
        units = 'dimensionless'
    elif color_parameter[:3].lower() == 'lat':
        data, units = sim.get_data('V2' if vector=='V' else 'B2', time=time)
        idx = sim._nearest(sim.theta.value, lat.to(u.rad).value)
        vtheta = data[:,idx]
        color_values = vtheta
    
    # get background data
    if background_var:
        b_data, b_units = sim.get_data(background_var, time=time)
        idx = sim._nearest(sim.theta.value, lat.to(u.rad).value)
        b_slice = b_data[:,idx]
    
        # continuity across phi zero-2pi
        dphi = np.diff(sim.phi).mean()
        wrp_phi = np.concatenate((sim.phi, sim.phi[-1:] + dphi))
        wrp_b_slice = np.concatenate((b_slice, b_slice[0:1, :]), axis=0)
    
    ################################   PLOT   #################################
    
    # make figure
    fig = plt.figure(figsize=(9,9), dpi=300)
    # polar plot options
    ax = fig.add_subplot(projection='polar')
    ax.set_rorigin(0)
    ax.set_rlabel_position(-42.5)
    ax.set_rticks([0.5, 1, 1.5, 2])
    ax.set_title('Solar wind {} field in the {} plane'.format(
        allowed_vector.get(vector)[-1], 
        'ecliptic' if lat == 90*u.deg else 'lat = {}'.format(lat.to_value(u.deg) - 90)), 
                 fontsize='x-large', pad=20)
    rlabels = ax.get_ymajorticklabels()
    for label in rlabels:
        label.set_fontsize('large')
        label.set_color('black')
    tlabels = ax.get_xmajorticklabels()
    for label in tlabels:
        label.set_fontsize('large')
    
    if background_var:
        # set colormap direction so background is mostly light
        if np.mean(b_slice) < np.median(b_slice):
            b_cmap = 'binary_r'
        else:
            b_cmap = 'binary'
        # plot background
        contour = ax.contourf(wrp_phi, sim.r, wrp_b_slice.transpose(), 128, cmap = b_cmap) 
        # color bar parameter
        bar = plt.colorbar(contour, format="{x:.2e}", shrink=0.5, pad=0.05, anchor=(0.0, 0.5))
        bar.ax.tick_params(labelsize=12)
        bar.set_label(label='{} [{}]'.format(background_var, b_units), 
                      fontsize='large', loc='center')
    
    # plot vector field
    scale = 13000000 if vector == 'V' else 300
    quiver = ax.quiver(phi, r, vx_masked, vy_masked, color_values, 
                       cmap=cmap, scale = scale,
                       width=0.005, headlength = 2, headaxislength = 2)
    
    # cbar parameters
    # cbar = plt.colorbar(quiver, location='bottom', format="{x:.2e}", pad=0.06)
    # cbar.ax.tick_params(labelsize=12)
    # tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    # cbar.locator = tick_locator
    # cbar.update_ticks()
    # cbar.set_label(label='{} [{}]'.format(color_parameter, units), 
    #                 fontsize='large', loc='center')
    
    if flow_paths:
        # unknown whether one or more flow paths
        try: # try multiple flow paths
            for coords in flow_paths:
                phi = []; r = []
                for coord in coords:
                    phi.append(coord[2].value)
                    r.append(coord[0].value)
                ax.plot(phi, r, color='k', linewidth=1)
        except TypeError: # single flow path
            phi = []; r = []
            for coord in flow_paths:
                phi.append(coord[2].value)
                r.append(coord[0].value)
            ax.plot(phi, r, color='k', linewidth=1)
        
    if flow_rate_markers:
        viridis = mpl.colormaps.get_cmap('viridis')
        colors = viridis(np.linspace(0,1,max([len(p) for p in flow_rate_markers])))
        
        try: # unknown whether one or more flow paths
            for coords in flow_rate_markers: # try multiple flow paths
                phi=[]; r=[]
                for coord in coords:
                    phi.append(coord[2].value)
                    r.append(coord[0].value)
                scatter = ax.scatter(phi, r, c=colors[:len(coords)], marker='o', s=15, zorder=2)
                labels = np.arange(0,max([len(p) for p in flow_rate_markers]),2)
        except TypeError: # single flow path
            phi = []; r = []
            for coord in flow_rate_markers:
                phi.append(coord[2].value)
                r.append(coord[0].value)
            scatter = ax.scatter(phi, r, c=colors[:len(flow_rate_markers)], marker='o', s=15, zorder=2)
            labels = np.arange(0,len(flow_rate_markers),2)
        
        # colorbar parameters
        colorbar = plt.colorbar(scatter, location='left', shrink=0.5)
        colorbar.set_ticks(np.linspace(0,1,len(labels)))
        colorbar.set_ticklabels(labels, fontsize='large')
        colorbar.set_label('Travel Time [Days]', fontsize='large')
        
        # For correct display if using flow_rate_markers:
            # comment cbar parameters for vector field (quiver)
            # uncomment code below
            # change background contour plot bar anchor to anchor=(0.0, 0.5)
        # cbar parameters
        cbar = plt.colorbar(quiver, location='bottom', format="{x:.2e}", pad=0.06)
        cbar.ax.tick_params(labelsize=12)
        tick_locator = mpl.ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label(label='{} [{}]'.format(color_parameter, units), 
                        fontsize='large', loc='center')
        
    return fig


def flow_to_spacecraft(times, sc1, sc2, sim_file_path, cmap='viridis'):
    # instantiate coordinates and simulation
    coords = Coordinates(times, [sc1, sc2])
    sim = Simulation(sim_file_path)
    
    dt = np.diff(coords.times_mjd).mean()
    
    # find flow paths from spacecraft
    flow_splines = {}
    for sc in coords.spacecraft:
        sim.get_cells_from_sc_coords(coords, sc, fixed_dt = dt)
        # check_sc_sim_consistency(coords, sim, spacecraft=[sc])
        flow_times = []; flow_coordinates = []
        phi_splines = []; r_splines= []; # theta_splines = []
        for time, idx in zip(sim.sc_cell_times[sc], sim.sc_cell_idxs[sc]):
            coord = [sim.r[idx[0]].value, sim.theta[idx[1]].value, sim.phi[idx[2]].value]
        # for time, sc_coords in zip(coords.times_mjd, coords.sc_coords[sc]):
        #     coord = [sc_coords.distance.au, sc_coords.lat.rad, sc_coords.lon.rad]
            dts, flow_coords, _, _ = sim.trace_flow(coord, max_r = 1.1)
            dts = np.array(dts)
            dts = time + dts
            flow_coords = np.array(flow_coords).transpose()
            flow_times.append(dts)
            flow_coordinates.append(flow_coords)
            
    
            # fit splines to flow paths for re-sampling
            phi_splines.append(CubicSpline(dts, flow_coords[2]))
            r_splines.append(CubicSpline(dts, flow_coords[0]))
            # theta_splines.append(CubicSpline(dts, flow_coords[1] - np.pi/2))
            
        flow_splines[sc] = [flow_times, phi_splines, r_splines] # , theta_splines]
    
        # print(flow_coordinates[0])
    
        # plt.plot(flow_times[0], phi_splines[0](flow_times[0]))
        # plt.show()
        # plt.plot(flow_times[0], phi_splines[0](flow_times[0]) - np.pi)
        # plt.show()
        
    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'Stereo-A']
    
    nsplines = len(flow_splines.get(sc1)[0])
    viridis = mpl.colormaps.get_cmap(cmap)
    colormap = viridis(np.linspace(0,1,20))

    fig = plt.figure(figsize=(8.5,8), dpi=300)
    ax = fig.add_subplot(projection='polar')
    
    def animate(i):
        
        ax.clear()
        
        ax.set_ylim(0,1.1)
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_rlabel_position(-42.5)
        
        count = str((coords.times[i]).iso)
        ax.text(4,1.6, '0000-00-00 00:00:00.000', color='w', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
        ax.text(4,1.6, count)
        
        # Plot spacecraft positions
        sc1_pos = ax.scatter(coords.sc_coords.get(sc1)[i].lon.rad, coords.sc_coords.get(sc1)[i].distance.au, c=colors[sc_index[sc1]], s=20)
        sc2_pos = ax.scatter(coords.sc_coords.get(sc2)[i].lon.rad, coords.sc_coords.get(sc2)[i].distance.au, c=colors[sc_index[sc2]], s=20)
        
        ax.legend([sc1_pos, sc2_pos], [labels[sc_index[sc1]], labels[sc_index[sc2]]], 
                  loc='lower right', bbox_to_anchor=(1.1,-0.1), markerscale=1, 
                  labelspacing=0.8, frameon=False)
        
        time_mjd = coords.times_mjd[i]
        
        # Plot flow path
        # for sc in coords.spacecraft:
        for n, flow_times, phi_spline, r_spline in zip(range(nsplines), flow_splines.get(sc1)[0], flow_splines.get(sc1)[1], flow_splines.get(sc1)[2]):
            if (time_mjd >= flow_times[0]) and (time_mjd <= flow_times[-1]):
                phi = phi_spline(time_mjd)
                ax.scatter(phi, r_spline(time_mjd), color = colormap[n%20], s=15, alpha=0.75, zorder=0)

        if i%20 == 0:
            print('{}% complete'.format(int(i/len(coords.times)*100)))
        
        plt.show()
        
    orbits = FuncAnimation(fig, animate, frames=np.arange(len(coords.times_mjd)), interval=100)
    print('Saving animation to .mp4 file. This may take several minutes.')
    orbits.save('./figures/Flow path {} to {}.mp4'.format(sc1, sc2),
                writer = 'ffmpeg', fps = 10)
    print('Save completed.')
    

def flow_from_spacecraft(coords, sc, sim_file_path, cmap='plasma'):
    # instantiate coordinates and simulation
    sim = Simulation(sim_file_path)
    
    dt = np.diff(coords.times_mjd).mean()
    
    # find flow paths from spacecraft
    sim.get_cells_from_sc_coords(coords, sc, fixed_dt = dt)
    # check_sc_sim_consistency(coords, sim, spacecraft=[sc])
    flow_times = []; phi_splines = []; r_splines = []; # theta_splines = []
    # for time, idx in zip(sim.sc_cell_times[sc], sim.sc_cell_idxs[sc]):
    #     coord = [sim.r[idx[0]].value, sim.theta[idx[1]].value, sim.phi[idx[2]].value]
    for time, sc_coords in zip(coords.times_mjd, coords.sc_coords[sc]):
        coord = [sc_coords.distance.au, sc_coords.lat.rad, sc_coords.lon.rad]
        dts, flow_coords, _, _ = sim.trace_flow(coord, max_r = 1.1)
        dts = np.array(dts)
        dts = time + dts
        flow_times.append(dts)
        flow_coords = np.array(flow_coords).transpose()
        # fit splines to flow paths for re-sampling
        phi_splines.append(CubicSpline(dts, flow_coords[2]))
        r_splines.append(CubicSpline(dts, flow_coords[0]))
        # theta_splines.append(CubicSpline(dts, flow_coords[1] - np.pi/2))
        
        
    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    sc_colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    sc_labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'Stereo-A']
    
    nsplines = len(r_splines)
    viridis = mpl.colormaps.get_cmap(cmap)
    colormap = viridis(np.linspace(0,1,20))

    fig = plt.figure(figsize=(8.5,8), dpi=300)
    ax = fig.add_subplot(projection='polar')
    
    def animate(i):
        
        ax.clear()
        
        ax.set_ylim(0,1.1)
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_rlabel_position(-42.5)
        
        count = str((coords.times[i]).iso)
        ax.text(4,1.6, '0000-00-00 00:00:00.000', color='w', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
        ax.text(4,1.6, count)
        
        # Plot spacecraft positions
        handles = []; labels = []
        for spacecraft in coords.spacecraft:
            sc_pos = ax.scatter(coords.sc_coords.get(spacecraft)[i].lon.rad, 
                                coords.sc_coords.get(spacecraft)[i].distance.au, 
                                c=sc_colors[sc_index[spacecraft]], s=20, marker='s')
            handles.append(sc_pos)
            labels.append(sc_labels[sc_index[spacecraft]])
            
        ax.legend(handles, labels,
                  loc='lower right', bbox_to_anchor=(1.1,-0.1), markerscale=1, 
                  labelspacing=0.8, frameon=False)
        
        time_mjd = coords.times_mjd[i]
        
        # Plot flow path
        # for sc in coords.spacecraft:
        for n, flow_time, phi_spline, r_spline in zip(range(nsplines), flow_times, phi_splines, r_splines):
            if (time_mjd >= flow_time[0]) and (time_mjd <= flow_time[-1]):
                phi = phi_spline(time_mjd)
                ax.scatter(phi, r_spline(time_mjd), color = colormap[n%20], s=10, marker='*', alpha=0.75, zorder=0)

        if i%20 == 0:
            print('{}% complete'.format(int(i/len(coords.times)*100)))
        
        plt.show()
        
    orbits = FuncAnimation(fig, animate, frames=np.arange(len(coords.times_mjd)), interval=80)
    print('Saving animation to .mp4 file. This may take several minutes.')
    orbits.save('./figures/Flow path from {} from sc coords.mp4'.format(sc),
                writer = 'ffmpeg', fps = 10)
    print('Saved to ./figures/Flow path from {}.mp4'.format(sc))
            
#############################  TIMESERIES PLOTS  #############################

def log_scaler(x, min_val=1, max_val=5):
    x_scaled = np.log(x/min_val) / np.log(max_val/min_val) + 1
    return x_scaled

def minmax_scaler(x, x_min, x_max, min_val=1, max_val=5):
    x_std = (x - x_min) / (x_max - x_min)
    x_scaled = x_std * (max_val - min_val) + min_val
    return x_scaled

def get_line_density(conj, ts, conj_times, sc_cell_times):
    bin_edges = conj_times[0::4]
    dens, _ = np.histogram(sc_cell_times, bins=bin_edges)
    bin_edges.pop(-1)
    dens_spline = interp1d(bin_edges, dens, bounds_error=False, fill_value='extrapolate')
    times = np.linspace(conj_times[0], conj_times[-1], 1000)
    lwidths = dens_spline(times)
    points = np.array([times, ts(times)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments, lwidths

def simple_timeseries(ts, times, variables, units, spacecraft, plot_pts=True, plot_pts_density=False):
    
    # plotting options
    global sc_params
    sc_params = {'bepi': ['BepiColombo', 'indianred'], 'earth': ['Earth', 'darkgreen'], 
                  'psp': ['PSP', 'slategrey'], 'so': ['Solar Orbiter', 'steelblue'], 
                  'sta': ['STEREO-A', 'sandybrown']}
    title = 'Timeseries for {}'.format(', '.join([sc_params.get(sc)[0] for sc in spacecraft]))
    
    if isinstance(spacecraft, str):
        spacecraft = [sc.strip() for sc in spacecraft.split(',')]
        
    # parse variables
    if isinstance(variables, str):
        variables = [var.strip() for var in variables.split(',')]
    
    # get times & values        
    ts_times = {}
    ts_values = {}
    ts_min = {}
    ts_max = {}
    for sc in spacecraft: 
        condition = (ts.data['spacecraft'] == sc) & (ts.data['times'] >= times[0].mjd) & (ts.data['times'] <= times[-1].mjd)
        subset = ts.data[condition]
        ts_times[sc] = list(subset['times'].values)
        
    for var in variables:
        sc_values = []
        min_values = []
        max_values = []
        for sc in spacecraft:
            condition = (ts.data['spacecraft'] == sc) & (ts.data['times'] >= times[0].mjd) & (ts.data['times'] <= times[-1].mjd)
            subset = ts.data[condition]
            values = list(subset[var].values)
            sc_values.append(values)
            min_values.append(min(values))
            max_values.append(max(values))
        ts_values[var] = sc_values
        ts_min[var] = min_values
        ts_max[var] = max_values
    
    # time labelling for timeseries plot
    time_labels = []
    time_ticks = []
    if len(ts_times[spacecraft[0]]) < 8:
        for i, time in enumerate(ts_times[spacecraft[0]]):
            time_ticks.append(time)
            time_labels.append(t.Time(time, format='mjd').iso[:10])
    else:
        for i, time in enumerate(ts_times[spacecraft[0]]):
            if i%(int(len(ts_times[spacecraft[0]])/4)) == 0:
                time_ticks.append(time)
                time_labels.append(t.Time(time, format='mjd').iso[:10])
        
    # figure set-up
    fig = plt.figure(figsize=(17,9), dpi=400)
    ax = []
    
    for i, var in enumerate(variables):
        
        # add as many subplots as there are variables
        ax.append(fig.add_subplot(len(variables),1,i+1))
        
        # configure axes
        ymin = min(ts_min[var])
        ymax = max(ts_max[var])
        tol = (ymax - ymin)*0.1
        ax[i].set_ylim(ymin - tol, ymax + tol)
        ax[i].set_xlim(times[0].mjd, times[-1].mjd) # crop spline edge effects
        ax[i].set_xticks(time_ticks, labels=[])
        ax[i].set_ylabel('{} [{}]'.format(var, units[i]), 
                      fontsize='large')
        
        # plot timeseries
        # if plot_pts_density:
        #     segments1, lwidths1 = get_line_density(conj, ts[conj.spacecraft[0]], conj_times_mjd, times_sc1)
        #     segments2, lwidths2 = get_line_density(conj, ts[conj.spacecraft[1]], conj_times_mjd, times_sc2)
        #     min_width = min(min(lwidths1), min(lwidths2))
        #     max_width = max(max(lwidths1), max(lwidths2))
        #     if max_width > 20:
        #         lwidths1 = log_scaler(lwidths1)
        #         lwidths2 = log_scaler(lwidths2)
        #     else:
        #         lwidths1 = minmax_scaler(lwidths1, min_width, max_width)
        #         lwidths2 = minmax_scaler(lwidths2, min_width, max_width)
        #     sc1_line = LineCollection(segments1, linewidths=lwidths1, color=sc_params.get(conj.spacecraft[0])[1])
        #     sc2_line = LineCollection(segments2, linewidths=lwidths2, color=sc_params.get(conj.spacecraft[1])[1])
        #     ax[i].add_collection(sc1_line)
        #     ax[i].add_collection(sc2_line)
        # else:
            
        for j, sc in enumerate(spacecraft):
            ax[i].plot(ts_times[sc], ts_values.get(var)[j], color=sc_params.get(sc)[1], label=sc_params.get(sc)[0])
        
        # if plot_pts:
        #     # plot data points
        #     ax[i].scatter(times_sc1, ts1, color = 'k', s=2)
        #     ax[i].scatter(times_sc2, ts2, color = 'k', s=2)
        
    ax[-1].set_xticks(time_ticks, labels=time_labels, fontsize='large')
    ax[-1].set_xlabel('Date', labelpad=10, fontsize='large')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), frameon=False, ncol=len(spacecraft), fontsize='large')
    fig.suptitle(title, fontsize='large')
    
    plt.savefig('./figures/time_series_plot')
    plt.show()
    
    return fig


def plot_multiple_timeseries(conjunction, timeseries, show_conjunctions=True, 
                             conjunction_category=['cone', 'parker spiral', 'quadrature', 'opposition']):
                
    plt.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'})
    time_support()
    
    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
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
    for i in range(len(timeseries.spacecraft)):            
        ax.append(fig.add_subplot(len(timeseries.spacecraft),1,i+1))
        ax[i].plot(times, timeseries.data[i], color=colors[sc_index.get(timeseries.spacecraft[i])], 
                   label=labels[sc_index.get(timeseries.spacecraft[i])])
        
        if show_conjunctions:
            conjunctions = conjunction.find_conjunctions(2, 
                                                         spacecraft_names=[timeseries.spacecraft[i]], 
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
        if i+1 == len(timeseries.spacecraft):
            ax[i].set_xlabel(r'$\mathrm{Date}$', fontsize='x-large', labelpad=10)
        else:
            plt.tick_params('x', labelbottom=False)
        #plt.savefig('./Figures/time_series_plot')
    return fig

def timeseries_reference(sc, time_interval, var, ts, sim):
    
    time_interval = t.Time(time_interval)
    
    # select data subset to plot
    c1 = ts.data['spacecraft'] == sc
    c2 = ts.data['times'] >= (time_interval[0].mjd - 1)
    c3 = ts.data['times'] <= (time_interval[-1].mjd + 1)
    data = ts.data[c1 & c2 & c3]
    data = data.reset_index(drop=True)
    # TODO: check times
    times = list(data['times'].values)
    # get timeseries data
    ts_values = list(data[var].values)
    # get sim data
    sim_values = sim.data[var]
    units = ts.units[np.where(np.array(ts.variables) == var)[0][0]]

    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    sc_colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    sc_labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    title = 'Solar Wind Radial Velocity Timeseries at {}'.format(sc_labels[sc_index[sc]])
       
    fig, (ax_ts, ax_sc) = plt.subplots(ncols=2, width_ratios=[17,10], dpi = 300)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    ax_sc.remove()
    ax_sc = fig.add_subplot(1, 2, 2, projection='polar')
    
    ax_ts.set_xlim(times[0], times[-1])
    ymin = min(data[var])
    ymax = max(data[var])
    tol = (ymax - ymin)*0.1
    ax_ts.set_ylim(ymin - tol, ymax + tol)
    ax_ts.set_xlabel('Date', labelpad=10)
    ax_ts.set_ylabel('{} [{}]'.format(var, units))
        
    # Time labelling for timeseries plot
    time_labels = []
    time_ticks = []
    if len(times) < 6:
        for time in times:
            time_ticks.append(time)
            time_labels.append(t.Time(time, format='mjd').iso[:10])
    else:
        for i, time in enumerate(times):
            if i%(int(len(times)/6)) == 0:
                time_ticks.append(time)
                time_labels.append(t.Time(time, format='mjd').iso[:10])
    
    ax_ts.set_xticks(time_ticks, labels=time_labels, fontsize='large')#, usetex=True)
    
    fig.suptitle(title, fontsize='large')
    
    def animate(i):
               
        ax_sc.clear()
        
        ax_sc.set_ylim(0,1.1)
        ax_sc.set_rticks([0.5, 1], labels=[0.5, 1], color='white')
        ax_sc.set_rlabel_position(-42.5)
        #ax_sc.set_title(title,fontsize='x-large', pad=20)
        
        count = 'Date: {}'.format(t.Time(times[i], format='mjd').iso[:10])
        ax_sc.text(1.1, 1.6, 'Date: 0000-00-00', color='w', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
        ax_sc.text(1.1, 1.6, count)
        
        ax_sc.set_rorigin(0)
        # continuity across phi zero-2pi
        var_slice = sim_values[:,sim._nearest(sim.theta.value, np.pi/2)]
        dphi = np.diff(sim.phi.value).mean()
        wrp_phi = np.concatenate((sim.phi.value, sim.phi[-1:].value + dphi))
        wrp_var_slice = np.concatenate((var_slice, var_slice[0:1, :]), axis=0)            
        
        
        #phi0 = omega.value*t.TimeDelta(conj.times[0] - coord.times[0]).to_value('hr') + 10/9*np.pi + coord.phi0
        #phi = (omega.value*t.TimeDelta(conj.times[i]-sim.times[0]).to_value('hr') + phi0)%(2*np.pi) # +8/9*np.pi?
        #phi0 = sim.omega*(times[0] - sim.sc_cell_times[0])*(1*u.day).to_value(u.s)
        phi = (sim.omega*(times[i]-times[0])*(1*u.day).to_value(u.s))%(2*np.pi)
        # plot sim
        contour = ax_sc.contourf(wrp_phi + phi, sim.r, wrp_var_slice.transpose(), 100, cmap = 'viridis')
        
        # Plot spacecraft path
        sc_pos = ax_sc.scatter(data['lon'][i], data['r'][i], c='white', s=20, marker='s', edgecolors = 'black', linewidth=0.2)

        ax_sc.legend([sc_pos], [sc_labels[sc_index.get(sc)]], frameon=False, 
                      loc='lower center', bbox_to_anchor=(1,-0.3), markerscale=1.5)
        
        ##########################  TIMESERIES PLOT  ##########################
        
        sc_ts, = ax_ts.plot([times[i], times[i+1]], [ts_values[i], ts_values[i+1]], color=sc_colors[sc_index.get(sc)])
                            # label=sc_[sc_index.get(sc)])
        
        if i == 1:
            cbar = plt.colorbar(contour, shrink=0.7, pad=0.1)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(label='{} [{}]'.format(var, units), fontsize='large', loc='center')
            ax_ts.scatter(times, ts_values, color = 'k', s=2, zorder=0)
            ax_ts.legend([sc_ts], [sc_labels[sc_index.get(sc)]], loc='upper right', frameon=False)#, bbox_to_anchor=(0.5, 1.1))
        
        if i%20 == 0:
            print(str(int(i/len(times)*100)) + '% complete.')
        
        plt.show()
        
    animated_plot = FuncAnimation(fig, animate, frames=np.arange(len(times)-1), interval=100)
    print('Generating animation. This may take several minutes...')
    animated_plot.save('./figures/{}_timeseries_{}.mp4'.format(var, sc), writer = 'ffmpeg', fps = 10)
    print('Animation saved to ./figures/{}_timeseries_{}.mp4'.format(var, sc))
    #print('Generating html5 video')
    #link = animated_plot.to_html5_video()
    #print(link)
    

############################  CONJUNCTIONS PLOTS  ############################

# Plot a given conjunction in 2D
def plot_conjunction2D(coords, Conjunctions, conj, idx=0, plot_simultaneous_conjs=True, plot_all_sc=True, ENLIL=False, sim=None, variable=None, save_fig=True):
    
    if not float(idx).is_integer():
        raise Exception('idx must be an integer.')
    idx = int(idx)
    
    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    
    time = conj.times[idx]
    conjs = [conj]
    if plot_simultaneous_conjs:
        conjunctions = Conjunctions.find_conjunctions(time=time, verbose=False)
        if conjunctions:
            for conj in conjunctions:
                conjs.append(conj)
    
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
            omega = (2*np.pi/25.38*(u.rad/((u.day).to(u.hour))))
            dt = t.TimeDelta(time - sim.times[0]).to_value('hr')
            angle = dt*omega.value*u.rad
        else:
            angle = None
        fig, ax = plot_ENLIL_slice(sim, variable, time=time, 
                                   lat=90*u.deg, cmap=my_cmap, rotate=angle)
    else:
        fig = plt.figure(figsize=(8,8), dpi=300)
        ax = fig.add_subplot(projection='polar')
    
    ax.set_ylim(0,1.1)
    ax.set_rticks([0.25, 0.5, 0.75, 1])
    ax.set_rlabel_position(-42.5)
    
    for conj in conjs:
        idx = coords._nearest([time.mjd for time in conj.times], time.mjd)
        if any(conj.isparker) == True:
            Rsun = const.equatorial_radius.to(u.au)
            day2s = 86400*u.s # seconds in a day
            lon_spiral = []
            counter = 0
            
            for pp in conj.parkerpairs:
    
                r = conj.sc_coords[pp][idx].distance
                lon = conj.sc_coords[pp][idx].lon
                swspeed = conj.swspeed[0][np.where(conj.parkerpairs == pp)[0][0]]
                
                r_spiral = np.linspace(0, 1.1, 50)
                lon_surface = lon.to(u.rad) + 2*np.pi*u.rad/(25.38*day2s)*(
                    r.to(u.au) - 2.5*Rsun)/(swspeed.to(u.au/u.s))
                lon_spiral.append(lon_surface.rad + 2*np.pi/(-25.38*day2s.value)*(
                    r_spiral - 2.5*Rsun.value)/(swspeed.to(u.au/u.s).value))
                
                ax.plot(lon_spiral[counter], r_spiral, c=colors[sc_index.get(pp)], linestyle='-.')
                counter += 1                
    
        spacecraft = list(set(conj.spacecraft).difference(set(conj.parkerpairs)))
        if spacecraft:
            x = []
            y = []
            
            # Positions of spacecraft in cartesian coordinates
            for sc in list(set(conj.spacecraft).difference(set(conj.parkerpairs))):    
                x.append(conj.sc_coords[sc][idx].distance.au*np.cos(conj.sc_coords[sc][idx].lon.rad))
                y.append(conj.sc_coords[sc][idx].distance.au*np.sin(conj.sc_coords[sc][idx].lon.rad))
            
            lon = [conj.sc_coords[sc][idx].lon.rad % (np.pi*2) for sc in conj.spacecraft]
                    
            # Parameters for shaded area
            theta = np.arange(min(lon), max(lon), 0.001)
            line = (-(y[1] - y[0]) / (x[1] - x[0]) * x[0] + y[0]) / (
                np.sin(theta) - ((y[1] - y[0]) / (x[1] - x[0]) * np.cos(theta)))
            ax.fill_between(theta, 0, line, facecolor = colors[5], alpha = 0.15)
            
    spacecraft = list(set([sc for conj in conjs for sc in conj.spacecraft]))
    
    # Plot!
    time_idx = coords._nearest([time.mjd for time in coords.times], time.mjd)
    
    for sc in spacecraft:
        ax.plot([coords.sc_coords[sc][time_idx].lon.rad, coords.sc_coords[sc][time_idx].lon.rad], 
                [0, coords.sc_coords[sc][time_idx].distance.au], c=colors[sc_index.get(sc)])
        
    if plot_all_sc:
        for sc in coords.spacecraft:
            ax.scatter(coords.sc_coords[sc][time_idx].lon.rad, coords.sc_coords[sc][time_idx].distance.au, 
                       c=colors[sc_index.get(sc)], label=labels[sc_index.get(sc)], s=15)
    else:
        for sc in spacecraft:
            ax.scatter(coords.sc_coords[sc][time_idx].lon.rad, coords.sc_coords[sc][time_idx].distance.au, 
                       c=colors[sc_index.get(sc)], label=labels[sc_index.get(sc)], s=15)
            
    
    if ENLIL:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.25,-0.08), markerscale=1, 
                  labelspacing=0.8, frameon=False, fontsize='medium')
        ax.text(5.23, 1.4, 'Date: {}'.format(str(conj.times[idx].iso)[0:10]), fontsize='large')
    else: # bbox_to_anchor changed from (-0.08,-0.15)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.15,-0.15), markerscale=1, 
                  labelspacing=0.8, frameon=False, fontsize='large')
        ax.text(5.25, 1.4, 'Date: {}'.format(str(conj.times[idx].iso)[0:10]), fontsize='large')
        
        
    tlabels = ax.get_xmajorticklabels()
    for label in tlabels:
        label.set_fontsize('large')
    rlabels = ax.get_ymajorticklabels()
    for label in rlabels:
        label.set_fontsize('large')
        
    title = 'Conjunctions between {} in {} coordinates'.format(
        ', '.join([labels[sc_index.get(sc)] for sc in spacecraft]), 
        coords.coordinate_name)
    ax.set_title(title, fontsize='x-large', multialignment='center', wrap=True, pad=20)
    
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
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    
    date = str((conj.start + idx*self.dt).iso)[0:10]
    
    title = 'Conjunctions between %s in %s coordinates' %(
        ', '.join([labels[sc_index.get(sc)] for sc in conj.spacecraft]), 
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
    
    # Positions of spacecraft in cartesian coordinates
    if self.coordinate_system == frames.HeliocentricInertial():
    
        for i in range(len(conj.spacecraft)):
            # Plot spacecraft paths
            xsat, ysat, zsat = self._spherical2cartesian(
                sat_coords[sc_index.get(conj.spacecraft[i])].distance.au, 
                np.pi + sat_coords[sc_index.get(conj.spacecraft[i])].lon.rad, 
                np.pi/2 + sat_coords[sc_index.get(conj.spacecraft[i])].lat.rad)
            ax.scatter(xsat, ysat, zsat, c=colors[sc_index.get(conj.spacecraft[i])], s=0.2)
            # Plot conjunctions
            x, y, z = self._spherical2cartesian(
                conj.coords[idx][i].distance.au, np.pi + conj.coords[idx][i].lon.rad, 
                np.pi/2 + conj.coords[idx][i].lat.rad)
            ax.plot([0, x], [0, y], [0, z], c=colors[sc_index.get(conj.spacecraft[i])], 
                    label=labels[sc_index.get(conj.spacecraft[i])])
            
        # Parameters for shaded area
        for i in range(0, len(conj.scpairs), 2):
            coords0 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[i])[0][0]]
            coords1 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[i+1])[0][0]]
            x0, y0, z0 = self._spherical2cartesian(
                coords0.distance.au, np.pi + coords0.lon.rad, np.pi/2 + coords0.lat.rad)
            x1, y1, z1 = self._spherical2cartesian(
                coords1.distance.au, np.pi + coords1.lon.rad, np.pi/2 + coords1.lat.rad)
            verts = [np.array([0,0,0]), np.array([x0,y0,z0]), np.array([x1,y1,z1])]
            ax.add_collection3d(Poly3DCollection([verts], facecolor = colors[5], alpha = 0.2))
    
    # Plot!
    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
        
        for i in range(len(conj.spacecraft)):
            # Plot spacecraft paths
            xsat, ysat, zsat = self._spherical2cartesian(
                sat_coords[sc_index.get(conj.spacecraft[i])].radius.au, 
                np.pi + sat_coords[sc_index.get(conj.spacecraft[i])].lon.rad, 
                np.pi/2 + sat_coords[sc_index.get(conj.spacecraft[i])].lat.rad)
            ax.scatter(xsat, ysat, zsat, c=colors[sc_index.get(conj.spacecraft[i])], s=0.2)
            # Plot conjunctions
            x, y, z = self._spherical2cartesian(
                conj.coords[idx][i].radius.au, np.pi + conj.coords[idx][i].lon.rad, 
                np.pi/2 + conj.coords[idx][i].lat.rad)
            ax.plot([0, x], [0, y], [0, z], c=colors[sc_index.get(conj.spacecraft[i])], 
                    label=labels[sc_index.get(conj.spacecraft[i])])
            
        # Parameters for shaded area
        for i in range(0, len(conj.scpairs), 2):
            coords0 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[i])[0][0]]
            coords1 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[i+1])[0][0]]
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
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
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
        
        sc_cell_times = self._get_sc_cell_times(sim_file_paths)
        for j in range(len(sc_cell_times)):
            sc_cell_times[j] = sc_cell_times[j].mjd
        
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
                    
        conjs = {2: self.twoscconjs, 3: self.threescconjs, 4: self.fourscconjs, 5: self.fivescconjs}
        
        if ENLIL:
            #TODO: add netcdf support
            idx = self._nearest(sc_cell_times, (self.starttime + self.dt*i).mjd)
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
            var_slice = var[:,self._nearest(theta, np.pi/2)]
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
                        # Positions of spacecraft in cartesian coordinates
                        for j in range(key):
                            x.append(conj.coords[idx][j].distance.au*np.cos(conj.coords[idx][j].lon.rad))
                            y.append(conj.coords[idx][j].distance.au*np.sin(conj.coords[idx][j].lon.rad))
                            
                        if any(conj.isparker) == True:
                            Rsun = const.equatorial_radius.to(u.au)
                            day2s = 86400*u.s # seconds in a day
                            lon_spiral = []
                            counter = 0
                            
                            for pp in conj.parkerpairs:

                                r = conj.coords[idx][np.where(conj.spacecraft == pp)[0][0]].distance
                                lon = conj.coords[idx][np.where(conj.spacecraft == pp)[0][0]].lon
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
                                        c=colors[sc_index.get(pp)], linestyle='-.')
                                counter += 1
                                
                    # different attribute names between HCI and HGC coordinates
                    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
                            # Positions of spacecraft in cartesian coordinates
                            for j in range(key):
                                x.append(conj.coords[idx][j].radius.au*np.cos(conj.coords[idx][j].lon.rad))
                                y.append(conj.coords[idx][j].radius.au*np.sin(conj.coords[idx][j].lon.rad))
                                
                            if 'parker spiral' in conj.category:
                                Rsun = const.equatorial_radius.to(u.au)
                                day2s = 86400 # seconds in a day
                                lon_spiral = []
                                counter = 0
                                
                                for pp in conj.parkerpairs:

                                    r = conj.coords[idx][np.where(conj.spacecraft == pp)[0][0]].radius
                                    lon = conj.coords[idx][np.where(conj.spacecraft == pp)[0][0]].lon
                                    
                                    r_spiral = np.linspace(0, 1.1, 50)
                                    lon_surface = lon + 2*np.pi*u.rad/(25.38*day2s.value)*(
                                        r.to(u.au) - 2.5*Rsun)/(conj.swspeed).to(u.au/u.s).value
                                    lon_spiral.append(
                                        lon_surface.rad + 2*np.pi/(-25.38*day2s)*
                                        (r_spiral - 2.5*Rsun.value)/
                                        (conj.swspeed*u.km).to(u.au).value)
                                    
                                    ax.plot(lon_spiral[counter], r_spiral, 
                                            c=colors[sc_index.get(pp)], linestyle='-.')
                                    counter += 1
                                
                    for j in range(0, len(conj.scpairs), 2):
                        coords0 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[j])[0][0]]
                        coords1 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[j+1])[0][0]]
                        verts = [np.array([0,0]), 
                                 np.array([coords0.lon.rad,coords0.distance.au]), 
                                 np.array([coords1.lon.rad,coords1.distance.au])]
                        ax.add_collection(
                            mpl.collections.PolyCollection([verts], facecolor = colors[5], alpha = 0.2))
                    
                    # Plot!
                    if self.coordinate_system == frames.HeliocentricInertial():
                        for j in range(len(conj.spacecraft)):
                            ax.plot([conj.coords[idx][j].lon.rad, conj.coords[idx][j].lon.rad], 
                                    [0, conj.coords[idx][j].distance.au], 
                                    c=colors[sc_index.get(conj.spacecraft[j])], 
                                    label=labels[sc_index.get(conj.spacecraft[j])])

                    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
                        for j in range(len(conj.spacecraft)):
                            ax.plot([conj.coords[idx][j].lon.rad, conj.coords[idx][j].lon.rad], 
                                    [0, conj.coords[idx][j].radius.au], 
                                    c=colors[sc_index.get(conj.spacecraft[j])], 
                                    label=labels[sc_index.get(conj.spacecraft[j])])
        
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
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
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
            for j in range(len(self.spacecraft)):
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
            for j in range(len(self.spacecraft)):
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
        
        conjs = {2: self.twoscconjs, 3: self.threescconjs, 4: self.fourscconjs, 5: self.fivescconjs}
        
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
                        # Positions of spacecraft in cartesian coordinates
                        for j in range(key):
                            x, y, z = self._spherical2cartesian(
                                conj.coords[idx][j].distance.au, 
                                np.pi + conj.coords[idx][j].lon.rad, 
                                np.pi/2 + conj.coords[idx][j].lat.rad)

                    # Parameters for shaded area
                    for j in range(0, len(conj.scpairs), 2):
                        coords0 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[j])[0][0]]
                        coords1 = conj.coords[idx][np.where(conj.spacecraft == conj.scpairs[j+1])[0][0]]
                        x0, y0, z0 = self._spherical2cartesian(
                            coords0.distance.au, np.pi + coords0.lon.rad, np.pi/2 + coords0.lat.rad)
                        x1, y1, z1 = self._spherical2cartesian(
                            coords1.distance.au, np.pi + coords1.lon.rad, np.pi/2 + coords1.lat.rad)
                        verts = [np.array([0,0,0]), np.array([x0,y0,z0]), np.array([x1,y1,z1])]
                        ax.add_collection3d(Poly3DCollection([verts], facecolor = colors[5], alpha = 0.2))
                     
                        
                    # Plot!
                    if self.coordinate_system == frames.HeliocentricInertial():
                        for j in range(len(conj.spacecraft)):
                            xsat, ysat, zsat = self._spherical2cartesian(
                                conj.coords[idx][j].distance.au, 
                                np.pi + conj.coords[idx][j].lon.rad, 
                                np.pi/2 + conj.coords[idx][j].lat.rad)
                            ax.plot([0, xsat], [0, ysat], [0, zsat], 
                                    c=colors[sc_index.get(conj.spacecraft[j])], 
                                    label=labels[sc_index.get(conj.spacecraft[j])])

                    if self.coordinate_system == frames.HeliographicCarrington(observer = 'earth'):
                        for j in range(len(conj.spacecraft)):
                            xsat, ysat, zsat = self._spherical2cartesian(
                                conj.coords[idx][j].radius.au, 
                                np.pi + conj.coords[idx][j].lon.rad, 
                                np.pi/2 + conj.coords[idx][j].lat.rad)
                            ax.plot([0, xsat], [0, ysat], [0, zsat], 
                                    c=colors[sc_index.get(conj.spacecraft[j])], 
                                    label=labels[sc_index.get(conj.spacecraft[j])])
                
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
    
    
    
################################  OTHER PLOTS  ###############################
    
def plot_spacecraft_position(coords, spacecraft, time):
    
    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    
    title = 'Position of {} in {} coordinates'.format(
        ', '.join([labels[sc_index.get(sc)] for sc in spacecraft]), 
        coords.coordinate_name)
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = fig.add_subplot(projection='polar')

    ax.set_ylim(0,1.1)
    #ax.set_rticks([0.25, 0.5, 0.75, 1])
    ax.set_rlabel_position(-42.5)
    ax.set_title(title, fontsize='x-large', multialignment='center', pad=20)
    
    for sc in spacecraft:
        time_idx = coords._nearest([time.mjd for time in coords.times], t.Time(time).mjd)
        ax.scatter(coords.sc_coords[sc][time_idx].lon.rad, coords.sc_coords[sc][time_idx].distance.au, 
                   c=colors[sc_index.get(sc)], label=labels[sc_index.get(sc)], s=10)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15,-0.15), markerscale=1, 
              labelspacing=0.8, frameon=False, fontsize='large')
    ax.text(5.25, 1.4, 'Date: {}'.format(str(t.Time(time).iso)[0:10]), fontsize='large')
        

def check_sc_sim_consistency(coords, sim, spacecraft=None):
    
    # TODO: phi coordinate behavior 0-->2pi gives erroneous sim coordinates
    
    # Plotting options
    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
    # colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
    labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
    
    if not spacecraft:
        spacecraft = sim.spacecraft
    
    for i, sc in enumerate(spacecraft):
        fig, (axr, axtheta, axphi) = plt.subplots(nrows=3, sharex=True, dpi=300)
        fig.set_figwidth(14)
        fig.set_figheight(8)
        
        dt = t.TimeDelta(np.nanmean(np.diff(coords.times_mjd)))
        
        ############  RADIUS  ############
        axr.set_title('R-coordinate for {}'.format(labels[sc_index.get(sc)]), fontsize='large')
        axr.set_ylabel('Heliocentric distance [AU]', fontsize='large')
        
        r = [coord.distance.au for coord in coords.sc_coords.get(sc)]
        sim_r = sim.r[sim.sc_cell_idxs.get(sc)[:,0]].value
        
        sc_curve = axr.scatter(coords.times_mjd, r, s=5, color = 'gold')
        sim_pts = axr.scatter(sim.sc_cell_times[sc], list(sim_r), s=2, c='k', edgecolors='none')
        axr.legend([sc_curve, sim_pts], ['S/C trajectory', 'Simulation cell center'], frameon=False)
        axr.set_xlim(coords.times_mjd[0], coords.times_mjd[-1])
        
        ############  THETA  ############
        axtheta.set_title('Theta-coordinate for {}'.format(labels[sc_index.get(sc)]), fontsize='large')
        axtheta.set_ylabel('Latitude [deg]', fontsize='large')
        
        theta = [coord.lat.rad + np.pi/2 for coord in coords.sc_coords.get(sc)]
        sim_theta = sim.theta[sim.sc_cell_idxs.get(sc)[:,1]].value
        
        sc_curve = axtheta.scatter(coords.times_mjd, np.degrees(theta), s=5, color = 'olivedrab')
        sim_pts = axtheta.scatter(sim.sc_cell_times[sc], np.degrees(list(sim_theta)), s=2, c='k', edgecolors='none')
        axtheta.legend([sc_curve, sim_pts], ['S/C trajectory', 'Simulation cell center'], frameon=False)
        axtheta.set_xlim(coords.times_mjd[0], coords.times_mjd[-1])
        
        ############  PHI  ############
        axphi.set_title('Phi-coordinate for {}'.format(labels[sc_index.get(sc)]), fontsize='large')
        axphi.set_xlabel('Times [mjd]', fontsize='large')
        axphi.set_ylabel('Carrington Longitude [deg]', fontsize='large')
        
        # # NO ROTATION
        # phi = []
        # for lon in coords.sc_coords.get(sc).lon.rad:
        #     if lon < 0:
        #         phi.append(lon + 2*np.pi)
        #     else:
        #         phi.append(lon)
        # sim_phi = sim.phi[sim.sc_cell_idxs.get(sc)[:,2]].value
        
        # ROTATION
        phi = []
        for j, lon in enumerate(coords.sc_coords.get(sc).lon.rad):
            if lon < 0:
                phi.append((lon + 2*np.pi - sim.omega*t.TimeDelta(coords.times[j]-coords.times[0]).to_value(u.s)) % (2*np.pi))
            else:
                phi.append((lon - sim.omega*t.TimeDelta(coords.times[j]-coords.times[0]).to_value(u.s)) % (2*np.pi))
        sim_phi = sim.phi[sim.sc_cell_idxs.get(sc)[:,2]].value
        
        
        #axphi.plot(times, phi, color = 'b')
        sc_curve = axphi.scatter(coords.times_mjd, np.degrees(phi), s=5, color = 'cornflowerblue')
        sim_pts = axphi.scatter(sim.sc_cell_times[sc], np.degrees(list(sim_phi)), s=2, c='k', edgecolors='none')
        axphi.legend([sc_curve, sim_pts], ['S/C trajectory', 'Simulation cell center'], frameon=False, loc='upper right')
        axphi.set_xlim(coords.times_mjd[0], coords.times_mjd[-1])
        
        plt.show()
        

# TODO: work in progress
def get_timeseries_information_density(conj, sim):
    
    omega = 2*np.pi*u.rad/(25.38*u.day)
    for sc in conj.spacecraft:
        r = [sim.r[idxs[0]].value for idxs in conj.sc_cell_idxs.get(sc)]
        theta = [sim.theta[idxs[1]].value for idxs in conj.sc_cell_idxs.get(sc)]
        phi = [sim.phi[idxs[2]].value for idxs in conj.sc_cell_idxs.get(sc)]
        sc_cell_times = [time.mjd for time in conj.sc_cell_times.get(sc)]
    
        # find optimal sampling rate based on S/C velocity through sim cells
        dr_dt = abs(np.diff(r))/t.TimeDelta(np.diff(sc_cell_times)).to_value('day')
        dtheta_dt = abs(np.diff(theta))/t.TimeDelta(np.diff(sc_cell_times)).to_value('day')
        dphi_dt = omega.value*np.diff(phi)/t.TimeDelta(np.diff(sc_cell_times)).to_value('day')
        pts = []
        
        for i in range(len(dr_dt)):
            if dr_dt[i] != 0 or dtheta_dt[i] != 0 or dphi_dt[i] != 0:
                pts.append(sc_cell_times[i])
                
        time_increments = pts - pts[0]
        bin_width = 1
        bins = int(time_increments[-1] / bin_width) + 1  # Calculate the number of bins

        plt.hist(time_increments, bins=bins, edgecolor='black')
        plt.xlabel('Days since {}'.format(conj.sc_cell_times.get(sc)[0].iso))
        plt.ylabel('Number of data points')
        plt.title('Histogram of information density for {}'.format(sc))
        plt.show()
        
def plot_sim_data_density(conj, sim):
    # no. of data points per day
    
    omega = 2*np.pi*u.rad/(25.38*u.day)
    for sc in conj.spacecraft:
        r = [sim.r[idxs[0]].value for idxs in conj.sc_cell_idxs.get(sc)]
        theta = [sim.theta[idxs[1]].value for idxs in conj.sc_cell_idxs.get(sc)]
        phi = [sim.phi[idxs[2]].value for idxs in conj.sc_cell_idxs.get(sc)]
        sc_cell_times = [time.mjd for time in conj.sc_cell_times.get(sc)]
    
        # find optimal sampling rate based on S/C velocity through sim cells
        dr_dt = abs(np.diff(r))/t.TimeDelta(np.diff(sc_cell_times)).to_value('day')
        dtheta_dt = abs(np.diff(theta))/t.TimeDelta(np.diff(sc_cell_times)).to_value('day')
        dphi_dt = omega.value*np.diff(phi)/t.TimeDelta(np.diff(sc_cell_times)).to_value('day')
        pts = []
        
        for i in range(len(dr_dt)):
            if dr_dt[i] != 0 or dtheta_dt[i] != 0 or dphi_dt[i] != 0:
                pts.append(sc_cell_times[i])
                
        time_increments = pts - pts[0]
        bin_width = 1
        bins = int(time_increments[-1] / bin_width) + 1  # Calculate the number of bins

        plt.hist(time_increments, bins=bins, edgecolor='black')
        plt.xlabel('Days since {}'.format(conj.sc_cell_times.get(sc)[0].iso))
        plt.ylabel('Number of data points')
        plt.title('Histogram of information density for {}'.format(sc))
        plt.show()
        
        # Plotting options
        # sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
        # colors = ['indianred', 'darkgreen', 'slategrey', 'steelblue', 'sandybrown', 'slategrey']
        # labels = ['BepiColombo', 'Earth', 'PSP', 'Solar Orbiter', 'STEREO-A']
        # title = 'Timeseries at %s, %s, %s, %s and %s' %(labels[0], labels[1], 
        #                                                 labels[2], labels[3], labels[4])
        
        # Figure set-up
        fig = plt.figure(figsize=(9,8), dpi=300)
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        
        ax2.plot(sc_cell_times, r, c='r')
        ax2.plot(sc_cell_times, theta, c='g')
        ax2.plot(sc_cell_times, phi, c='b')
        
        sc_cell_times.pop(-1)
        
        ax1.scatter(sc_cell_times, dr_dt, c='r', s=10, alpha=0.2)
        ax1.scatter(sc_cell_times, dtheta_dt, c='g', s=5, alpha=0.2)
        ax1.scatter(sc_cell_times, dphi_dt, c='b', s=5, alpha=0.2)
        
        ax1.set_title('gradients for {}'.format(sc))
        ax2.set_title('values for {}'.format(sc))
        
        plt.show()
