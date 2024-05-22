# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:17:53 2023

@author: Zoe.Faes
"""

import numpy as np
import pandas as pd
import spiceypy as spice
import astropy.time as t
import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
from coordinates import Coordinates
from urllib import request
from tslearn import metrics
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from features import get_dtw_matrix

pd.set_option('display.max_columns', None)

def get_icmes():
    
    url = 'https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    icmecat = pd.read_csv(url)
    
    icmes = {'sc': [], 'start': [], 'end': [], 'duration': []}
    for index, row in icmecat.iterrows():
        if row['sc_insitu'] in ['SolarOrbiter', 'PSP', 'STEREO-A'] and t.Time(row['icme_start_time']) < t.Time("2023-01-01 01:00:00.000") and t.Time(row['icme_start_time']) > t.Time('2021-12-31 18:00:00.000'):
            icmes['sc'].append(row['sc_insitu'])
            icmes['start'].append(t.Time(row['icme_start_time']))
            icmes['end'].append(t.Time(row['mo_end_time']))
            icmes['duration'].append(t.TimeDelta(float(row['icme_duration'])*u.hr))
            
    return pd.DataFrame(data=icmes)



# solo & psp vars: ['B', 'BN', 'BR', 'BT', 'ProtonSpeed', 'VN', 'VR', 'VT', 'protonDensity', 'protonTemp']
# sta vars: ['B', 'BN', 'BR', 'BT', 'plasmaSpeed', 'plasmaDensity', 'plasmaTemp']
def get_insitu_data(start='2022-01-01 12:00:00.000', end='2022-12-31 12:00:00.000', spacecraft=None):
    trange = attrs.Time(start, end)
    datasets = {'SolarOrbiter': attrs.cdaweb.Dataset('SOLO_COHO1HR_MERGED_MAG_PLASMA'), 'PSP': attrs.cdaweb.Dataset('PSP_COHO1HR_MERGED_MAG_PLASMA'), 'STEREO-A': attrs.cdaweb.Dataset('STA_COHO1HR_MERGED_MAG_PLASMA')}
    data = {}
    for sc, dataset in datasets.items():    
        result = Fido.search(trange, dataset)
        downloaded_files = Fido.fetch(result)
        
        data[sc] = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
    # data['STEREO-A'].peek(columns=['B', 'BN', 'BR', 'BT'])
    # data['STEREO-A'].peek(columns=['PlasmaSpeed'])
    # data['STEREO-A'].peek(columns=['plasmaDensity'])
    # data['STEREO-A'].peek(columns=['plasmaTemp'])
        
    return data

# icmes = get_icmes()
dataset = get_insitu_data(start='2022-01-01 12:00:00.000', end='2022-12-31 12:00:00.000')

df = dataset['SolarOrbiter']
so_v = df[['VR']]
so_data = df[['B', 'BN', 'BR', 'BT', 'VR', 'protonDensity', 'protonTemp']]
df = dataset['PSP']
psp_v = df[['VR']]
psp_data = df[['B', 'BN', 'BR', 'BT', 'VR', 'protonDensity', 'protonTemp']]
df = dataset['STEREO-A']
sta_v = df[['plasmaSpeed']]
sta_data = df[['B', 'BN', 'BR', 'BT', 'plasmaSpeed', 'plasmaDensity', 'plasmaTemp']]

# ax1 = so_v.plot(figsize=(12,1))
# ax1.set_ylabel('km/s')
# ax1.legend().set_visible(False)
# ax1.set_yticks([300,800])
# ax2 = psp_v.plot(figsize=(12,1))
# ax2.set_ylabel('km/s')
# ax2.legend().set_visible(False)
# ax2.set_yticks([300,800])
# ax3 = sta_v.plot(figsize=(12,1))
# ax3.set_ylabel('km/s')
# ax3.legend().set_visible(False)
# ax3.set_yticks([300,800])

# so_plot = so_data.plot(subplots = [['B', 'BN', 'BR', 'BT'], ['VR'], ['protonDensity'], ['protonTemp']], sharex=True, figsize=(12, 6))
# psp_plot = psp_data.plot(subplots = [['B', 'BN', 'BR', 'BT'], ['VR'], ['protonDensity'], ['protonTemp']], sharex=True, figsize=(12, 6))
# sta_plot = sta_data.plot(subplots = [['B', 'BN', 'BR', 'BT'], ['plasmaSpeed'], ['plasmaDensity'], ['plasmaTemp']], sharex=True, figsize=(12, 6))

#(0, 8760, 120)
# so_psp = []
# so_sta = []
# psp_sta = []
# for i in range(0, 8760, 48):
#     so_sub = so_data.iloc[i:i+120].values
#     psp_sub = psp_data.iloc[i:i+120].values
#     sta_sub = sta_data.iloc[i:i+120].values
#     sim1 = []
#     sim2 = []
#     sim3 = []
#     so_bool = True
#     psp_bool = True
#     sta_bool = True
#     for j in range(7):
#         so = so_sub[:,j]
#         psp = psp_sub[:,j]
#         sta = sta_sub[:,j]
#         if np.sum(np.isnan(so)) == len(so):
#             so_bool = False
#         if np.sum(np.isnan(psp)) == len(psp):
#             psp_bool = False
#         if np.sum(np.isnan(sta)) == len(sta):
#             sta_bool = False
#         if so_bool and psp_bool:
#             path_so_psp, sim_so_psp = metrics.dtw_path(so, psp)
#             sim1.append(sim_so_psp)
#         if so_bool and sta_bool:
#             path_so_sta, sim_so_sta = metrics.dtw_path(so, sta)
#             sim2.append(sim_so_sta)
#         if psp_bool and sta_bool:
#             path_psp_sta, sim_psp_sta = metrics.dtw_path(psp, sta)
#             sim3.append(sim_psp_sta)
#     so_psp.append(np.nanmean(sim1))
#     so_sta.append(np.nanmean(sim2))
#     psp_sta.append(np.nanmean(sim3))
    
# #print(so_sta)

# for comp in [so_psp, so_sta, psp_sta]:
#     plt.scatter(range(len(comp)), comp)
#     plt.yscale('log')
#     #plt.ylim([0,500])
#     plt.xlim([0,183])
#     num_ticks = 13
#     x_tick_positions = np.linspace(0, len(comp) - 1, num_ticks, dtype=int)
#     plt.xticks(x_tick_positions)
#     plt.show()
    
#########################################################
#                           MATRIX PLOT
#########################################################

# 22 - 29 March solo/psp - 22/03 = 1944 - 29/03 = 2112
# 28 Aug - 11 Sep solo/sta - 28/08 = 5760 - 11/09 = 6096
# 29 Nov - 15 Dec solo/sta - 29/11 = 7992 - 15/12 = 8376

# get timeseries
var = 'D'
so_d = so_data[['protonDensity']]
sta_d = sta_data[['plasmaDensity']]

var = 'T'
so_t = so_data[['protonTemp']]
sta_t = sta_data[['plasmaTemp']]

ts1 = so_t.iloc[5760:6095].values
# ts2 = psp_v.iloc[1944:2112].values
ts2 = sta_t.iloc[5760:6096].values

ts3 = ts1.flatten()
ts4 = ts2.flatten()

score, path = get_dtw_matrix(ts3, ts4, 
                             ts_labels=['Solar Orbiter - {}'.format(var), 'STEREO-A - {}'.format(var)], 
                             times=['28 Aug. 2022', '11 Sep. 2022'])


# axes1 = subset1.plot(subplots = [['B', 'BN', 'BR', 'BT'], ['VN', 'VR', 'VT'], ['protonDensity'], ['protonTemp']], sharex=True, figsize=(12, 6))
# axes1[0].set_ylabel('nT')
# axes1[1].set_ylabel('km/s')
# axes1[2].set_ylabel('N/cm3')
# axes1[3].set_ylabel('K')
# axes1[0].legend(loc=1)
# axes1[1].legend(loc=1)
# axes1[2].legend(loc=1)
# axes1[3].legend(loc=1)

# for sc, data in dataset.items():
#     nan_before = data.isnull().any(axis=1).sum()
#     icmes_subset = icmes[icmes['sc']==sc]
#     for index, row in icmes_subset.iterrows():
#         for idx, r in data.iterrows():
#             if t.Time(idx) >= row['start'] and t.Time(idx) <= row['end']:
#                 data.loc[idx, data.columns] = np.nan
#     nan_after = data.isnull().any(axis=1).sum()
#     print(nan_before, nan_after)
    
# print(dataset['SolarOrbiter'].columns)

# df = dataset['SolarOrbiter']
# subset1 = df[['B', 'BN', 'BR', 'BT', 'VN', 'VR', 'VT', 'protonDensity', 'protonTemp']]
# axes1 = subset1.plot(subplots = [['B', 'BN', 'BR', 'BT'], ['VN', 'VR', 'VT'], ['protonDensity'], ['protonTemp']], sharex=True, figsize=(12, 6))
# axes1[0].set_ylabel('nT')
# axes1[1].set_ylabel('km/s')
# axes1[2].set_ylabel('N/cm3')
# axes1[3].set_ylabel('K')
# axes1[0].legend(loc=1)
# axes1[1].legend(loc=1)
# axes1[2].legend(loc=1)
# axes1[3].legend(loc=1)

# df = dataset['STEREO-A']
# subset = df[['B', 'BN', 'BR', 'BT', 'plasmaSpeed', 'plasmaDensity', 'plasmaTemp']]
# axes = subset.plot(subplots = [['B', 'BN', 'BR', 'BT'], ['plasmaSpeed'], ['plasmaDensity'], ['plasmaTemp']], sharex=True, figsize=(12, 6))
# axes[0].set_ylabel('nT')
# axes[1].set_ylabel('km/s')
# axes[2].set_ylabel('N/cm3')
# axes[3].set_ylabel('K')

# test = get_insitu_data(start='2022-09-01 12:00:00.000', end='2022-09-01 14:00:00.000')
# solo_test = test['SolarOrbiter']
# print(solo_test)
# sub = solo_test[['VN', 'VR', 'VT']]
# print(np.shape(sub))

# coordinates = Coordinates(spacecraft_names=['Solar Orbiter'], times=['2022-09-01 00:00:00.000', '2022-09-01 01:00:00.000'])
# coords = coordinates.get_sc_coordinates(spacecraft='so', times=['2022-09-01 00:00:00.000', '2022-09-01 01:00:00.000'])
# print(coords)
# print(coords[0].distance.to(u.au))

def reproject_3D_data(data, old_frame, new_frame, spice_kernels):
    
    # load required SPICE kernels
    spice.furnsh(spice_kernels)
    
    new_values = np.empty(np.shape(data))
    
    for i in range(len(data)):
        et = spice.str2et(str(data.index[i])) # get time
        mat = spice.pxform(old_frame, new_frame, et) # get transition matrix
        new_values[i] = spice.mxv(mat, data.values[i]) # multiply
        
    data.iloc[:,:] = new_values
    
    return data
        
        
def download_spice_kernels(spacecraft_names):
    
    spacecraft = _parse_sc_names(spacecraft_names)
    
    # Define the remote file to retrieve
    meta_kernel_url = 'https://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/mk/solo_ANC_soc-pred-mk.tm'
    # Define the local filename to save data
    local_file = './spice_kernels/solo_ANC_soc-pred-mk.tm'
    # Download remote and save locally
    request.urlretrieve(meta_kernel_url, local_file)
    
    return local_file
    
    # kernel_urls = [
    #     "ck/solo_ANC_soc-sc-fof-ck_20180930-21000101_V03.bc",
    #     "ck/solo_ANC_soc-stix-ck_20180930-21000101_V03.bc",
    #     "ck/solo_ANC_soc-flown-att_20221011T142135-20221012T141817_V01.bc",
    #     "fk/solo_ANC_soc-sc-fk_V09.tf",
    #     "fk/solo_ANC_soc-sci-fk_V08.tf",
    #     "ik/solo_ANC_soc-stix-ik_V02.ti",
    #     "lsk/naif0012.tls",
    #     "pck/pck00010.tpc",
    #     "sclk/solo_ANC_soc-sclk_20231015_V01.tsc",
    #     "spk/de421.bsp",
    #     "spk/solo_ANC_soc-orbit-stp_20200210-20301120_280_V1_00288_V01.bsp",
    #     ]
    # kernel_urls = [f"http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/{url}"
    #                for url in kernel_urls]
        
        
        
def _parse_sc_names(spacecraft_names):
    # parse specified spacecraft names and store in self.bodies
    if type(spacecraft_names) == str:
        spacecraft_names = spacecraft_names.split(',')
    
    allowed_names = {'so': ['so', 'solar orbiter', 'solo'], 
                     'psp': ['psp', 'parker solar probe', 'parker probe', 'solar probe plus'], 
                     'bepi': ['bepi', 'bepicolombo', 'bepi colombo'], 
                     'sta': ['sa', 'sta', 'stereo-a', 'stereo a', 'stereoa'], 
                     'earth': ['earth', 'erde', 'aarde', 'terre', 'terra', 
                               'tierra', 'blue dot', 'home', 'sol d']}
    
    sc_names = []
    for name in spacecraft_names:
        name = name.strip()
        no_match = True
        for key, names in allowed_names.items():
            if name.lower() in names:
                sc_names.append(key)
                no_match = False
        if no_match:
            raise Exception('Invalid spacecraft name. Specify choice with '\
                            'a string containing the name of a spacecraft,'\
                            ' for example: \'Solar Orbiter\' or \'SolO\'.'\
                            ' spacecraft other than Solar Orbiter, Parker'\
                            ' Solar Probe, BepiColombo and STEREO-A are'\
                            ' not yet supported -- got \'{}\''
                            .format(name))
                
    return sc_names
        
def kpool(meta_kernel):

    # Assign the path name of the meta kernel to META.
    META = meta_kernel

    # Load the meta kernel then use KTOTAL to interrogate the SPICE
    # kernel subsystem.
    spice.furnsh(META)

    count = spice.ktotal('ALL');
    print('Kernel count after load: {0}\n'.format(count))

    # Loop over the number of files; interrogate the SPICE system
    # with spiceypy.kdata for the kernel names and the type.
    # 'found' returns a boolean indicating whether any kernel files
    # of the specified type were loaded by the kernel subsystem.
    # This example ignores checking 'found' as kernels are known
    # to be loaded.
    for i in range(0, count):
        [ file, type, source, handle] = spice.kdata(i, 'ALL');
        print( 'File   {0}'.format(file) )
        print( 'Type   {0}'.format(type) )
        print( 'Source {0}\n'.format(source) )

    # Now unload the meta kernel. This action unloads all
    # files listed in the meta kernel.
    spice.unload( META )

    # Check the count; spiceypy should return a count of zero.
    count = spice.ktotal( 'ALL');
    print( 'Kernel count after meta unload: {0}'.format(count))

    # Done. Unload the kernels.
    spice.kclear
    
# kpool(download_spice_kernels('solo'))