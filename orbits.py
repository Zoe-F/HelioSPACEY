# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:49:55 2022

First draft of a script to find and plot the orbits of Solar Orbiter,
Parker Solar Probe, BepiColombo and STEREO-A from SPICE kernels

@author: Zoe.Faes
"""

# Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import astropy.units as u
import astropy.time as t
from sunpy.coordinates import HeliocentricInertial
import astrospice


# Available kernels
print(astrospice.registry)

############################ GET COORDINATES #################################

# Get kernels
so_kernels = astrospice.registry.get_kernels('solar orbiter', 'predict')
so_kernel = so_kernels[0]
#print(so_kernel.bodies)
#so_coverage = so_kernel.coverage('SOLAR ORBITER')

psp_kernels = astrospice.registry.get_kernels('psp', 'predict')
psp_kernel = psp_kernels[0]
#print(psp_kernel.bodies)
#psp_coverage = psp_kernel.coverage('SOLAR PROBE PLUS')

sa_kernels = astrospice.registry.get_kernels('stereo-a', 'predict')
sa_kernel = sa_kernels[0]
#print(sa_kernel.bodies)
#sa_coverage = sa_kernel.coverage('STEREO AHEAD')

bepi_kernel = astrospice.kernel.SPKKernel('bc_mpo_fcp_00129_20181020_20251101_v01.bsp')
#print(bepi_kernel.bodies)
#bepi_coverage = bepi_kernel.coverage('BEPICOLOMBO MPO')

# Specify time range and increment to generate coordinates for
    # SO starttime = '2020-02-11', endtime = '2030-11-20'
starttime = '2020-02-11'
endtime = '2023-01-01'
dt = t.TimeDelta(24*u.hour)
times = t.Time(np.arange(t.Time(starttime), t.Time(endtime), dt))

# Get coordinates for times
so_coords = astrospice.generate_coords('SOLAR ORBITER', times)
psp_coords = astrospice.generate_coords('SOLAR PROBE PLUS', times)
sa_coords = astrospice.generate_coords('STEREO AHEAD', times)
bepi_coords = astrospice.generate_coords('BEPICOLOMBO MPO', times)

# Coordinate transform
HCI = HeliocentricInertial()
so_coords = so_coords.transform_to(HCI)
psp_coords = psp_coords.transform_to(HCI)
sa_coords = sa_coords.transform_to(HCI)
bepi_coords = bepi_coords.transform_to(HCI)


#################################  PLOT  #####################################

# Plot figure?
figure = True
# Make animation?
animation = True

# Plot title
title = 'Satellite positions from %s to %s' %(starttime, endtime)

# Plotting options
colors = ['steelblue', 'slategrey', 'indianred', 'darkgreen']
labels = ['Solar Orbiter', 'PSP', 'BepiColombo', 'Stereo-A']

mpl.rc('text', usetex = True)

if figure==True:
    fig1 = plt.figure(figsize=(8,8), dpi=400)
    ax1 = fig1.add_subplot(projection='polar')
    ax1.set_ylim(0,1.1)
    ax1.set_rticks([0.25, 0.5, 0.75, 1])
    ax1.set_rlabel_position(-42.5)
    ax1.set_title(title,fontsize='x-large')    
    
    # Thing to plot
    ax1.scatter(so_coords.lon.to(u.rad), so_coords.distance.to(u.au), c=colors[0], label=labels[0], s=2)
    ax1.scatter(psp_coords.lon.to(u.rad), psp_coords.distance.to(u.au), c=colors[1], label=labels[1], s=2)
    ax1.scatter(bepi_coords.lon.to(u.rad), bepi_coords.distance.to(u.au), c=colors[2], label=labels[2], s=2)
    ax1.scatter(sa_coords.lon.to(u.rad), sa_coords.distance.to(u.au), c=colors[3], label=labels[3], s=2)
    
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15,1.05), markerscale=2, labelspacing=0.8, frameon=False)
    
    plt.show()

if animation==True:

    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = fig.add_subplot(projection='polar')
    ax.set_ylim(0,1.1)
    ax.set_rticks([0.25, 0.5, 0.75, 1])
    ax.set_rlabel_position(-42.5)
    ax.set_title(title,fontsize='x-large')
        
    def animate(i):
        
        count = 'Date: %s' %str((t.Time(starttime) + dt*i).iso)[0:10]
        ax.text(4,1.6, 'Date: 0000-00-00', color='w', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
        ax.text(4,1.6, count)
        
        # Things to plot
        so = ax.scatter(so_coords.lon.to(u.rad)[i], so_coords.distance.to(u.au)[i], c=colors[0], s=2)
        psp = ax.scatter(psp_coords.lon.to(u.rad)[i], psp_coords.distance.to(u.au)[i], c=colors[1], s=2)
        bepi = ax.scatter(bepi_coords.lon.to(u.rad)[i], bepi_coords.distance.to(u.au)[i], c=colors[2], s=2)
        sa = ax.scatter(sa_coords.lon.to(u.rad)[i], sa_coords.distance.to(u.au)[i], c=colors[3], s=2)

        if i == 0:
            ax.legend([so, psp, bepi, sa], [labels[0], labels[1], labels[2], labels[3]], loc='lower right', bbox_to_anchor=(1.1,-0.1), markerscale=2, labelspacing=0.8, frameon=False)

        plt.show()
        
    orbits = FuncAnimation(fig, animate, frames=np.arange(len(times)), interval=20)
    print('Saving animation to .mp4 file. This may take several minutes.')
    print('Estimated time for timespan of 1 year with original parameters is 20 minutes.')
    orbits.save('./Animations/Orbits.mp4', writer = 'ffmpeg', fps = 30)
    print('Save completed.')
    #print('Generating html5 video')
    #link = orbits.to_html5_video()
    #print(link)
    
