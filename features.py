# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:50:00 2023

@author: Zoe.Faes
"""


# functions: preprocessing.normalize() vs preprocessing.MinMaxScaler(), balance dataset

# feature extraction methods for normalised timeseries

# lag
# Pearson correlation
# some distance functions
# difference
# descriptive statistics of the difference
# rho*V^2

# map distributions of features for different conjunctions and find which are most distinct - best indicators

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.time as t
import matplotlib.pyplot as plt
import seaborn as sns
import dill as pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, correlate
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from tslearn import metrics
import warnings

def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the warning category and message
    return '\n{}: {} \n'.format(category.__name__, msg)

file_names = ['Carolina_Almeida_011523_SH_1_20200701_20250701_6h',
              'Daniel_Phoenix_080621_SH_1_20200701_20250701_6h',
              'Daniel_Verscharen_101420_SH_1_20200701_20250701_6h',
              'Hyunjin_Jeong_050422_SH_3_20200701_20250701_6h',
              'Jihyeon_Son_090422_SH_4_20200701_20250701_6h',
              'limei_yan_032322_SH_1_20200701_20250701_6h',
              'limei_yan_032422_SH_1_20200701_20250701_6h',
              'limei_yan_032522_SH_2_20200701_20250701_6h',
              'limei_yan_032522_SH_3_20200701_20250701_6h',
              'Manuel_Grande_062021_SH_1_20200701_20250701_6h',
              'MariaElena_Innocenti_111020_SH_1_20200701_20250701_6h',
              'Michael_Terres_110920_SH_1_20200701_20250701_6h',
              'Michael_Terres_111020_SH_1_20200701_20250701_6h',
              'Ou_Chen_081721_SH_2_20200701_20250701_6h',
              'Peng_Shaoy_052822_SH_1_20200701_20250701_6h',
              'Peng_Shaoy_052822_SH_2_20200701_20250701_6h',
              'Peng_Shaoy_052822_SH_3_20200701_20250701_6h',
              'Qingbao_He_112022_SH_1_20200701_20250701_6h',
              'Qingbao_He_112022_SH_2_20200701_20250701_6h',
              'Sanchita_Pal_041621_SH_1_20200701_20250701_6h',
              'Zoe_Faes_101922_SH_1_20200701_20250701_6h']

file_paths = ['./Timeseries/{}.pickle'.format(name) for name in file_names]

label_labelling = {'non_conj': 0, 'cone': 1, 'quadrature': 2, 'opposition': 3, 'parker spiral': 4}


def find_lag(times, ts_x, ts_y, conj, plot_cc=False, plot_lag=False):
    
    npts = len(times)
    
    warnings.filterwarnings("error")
    
    lags = []
    for x, y in zip(ts_x, ts_y):
        pccx = []
        pccy = []
        for j in range(npts-round(npts/10)): # prevent divide by zero due to std. dev. computation from diag(cov) of numpy's corrcoef Spearman function due to duplicates
            try:
                pccx.append(x.corr(y.shift(j), method='spearman'))
                pccy.append(y.corr(x.shift(j), method='spearman'))
            except RuntimeWarning:
                print(j, npts, x.name, y.name)
                # pass
        lag = []
        best_direction = []
        for pcc in [pccx, pccy]:
            # peaks = find_peaks(pcc, height=0)
            # try:
            #     best = np.array([(npts - peaks[0][p])*peaks[1].get('peak_heights')[p] for p in range(len(peaks[0]))])
            #     lag.append(peaks[0][best.argmax()])
            #     best_direction.append(best.argmax())
            # except ValueError:
            lag.append(np.array(pcc).argmax())
            best_direction.append(1/(1+np.array(pcc).argmax()))
            peaks = False
        pcc = [pccx, pccy][np.array(best_direction).argmax()]
        lags.append(lag[np.array(best_direction).argmax()])
        
        if plot_cc:
            plt.figure()
            plt.plot(range(npts-round(npts/10)), pcc)
            plt.scatter(lags[-1], max(pcc), marker='x', color='orange')
            plt.title("Cross-correlation for {}".format(x.name))
            
        if plot_lag:
            sc_params = {'bepi': ['BepiColombo', 'indianred'], 'earth': ['Earth', 'darkgreen'], 
                          'psp': ['PSP', 'slategrey'], 'so': ['Solar Orbiter', 'steelblue'], 
                          'sta': ['STEREO-A', 'sandybrown']}
            xp = [x, y][np.array(best_direction).argmax()]
            yp = [x, y][np.array(best_direction).argmin()]
            plt.figure()
            plt.plot(range(npts), xp.values, color='b')
            shifted_series = yp.shift(lags[-1])
            plt.plot(range(npts), shifted_series.values, color='r') 
            
            title = '{} in a {} conjunction - var: {} - lag: {} hrs'.format(
                ' and '.join([sc_params.get(sc)[0] for sc in conj.spacecraft]), 
                conj.label[0], x.name[:-2], round((t.Time(times[lags[-1]]) - t.Time(times[0])).to_value('hr')))
        
            plt.title(title)
    warnings.resetwarnings() 
        
    lag = round(np.mean(lags))
    lag_time = t.Time(times[lag]) - t.Time(times[0])
    
    if lag_time > 500*u.hr:
        print('lag is greater than 500 hours.')
            
    return lag, lag_time



def find_lag2(conj, plot=True, check_plot=False):
    
    try:
        merged = conj.timeseries['merged']
    except KeyError:
        merge_timeseries(conj)
        merged = conj.timeseries['merged']

    lags = {}
    for var in conj.ts_units.keys():
        pcc = []
        npts = len(merged[str(var) + "_x"])
        for i in range(npts-1):
            pcc.append(merged[str(var + "_y")].corr(merged[str(var + "_x")].shift(i)))
        lags[var] = np.array(pcc).argmax()
        peaks = find_peaks(pcc, height=0)
        try:
            best = np.array([(npts - peaks[0][p])*peaks[1].get('peak_heights')[p] for p in range(len(peaks[0]))])
            lags[var] = peaks[0][best.argmax()]
            # TODO: get rid of bias for cones and do cc both ways, find highest correlation
        except ValueError:
            lags[var] = np.array(pcc).argmax()
            peaks = False
            if plot:
                plt.figure()
                plt.plot(range(len(pcc)), pcc)
                plt.scatter(lags[var], max(pcc), marker='x', color='orange')
                plt.title("Cross-correlation for {}".format(var))
        if plot and peaks:
            plt.figure()
            plt.plot(range(npts-1), pcc)
            plt.scatter(lags[var], pcc[lags[var]], marker='x', color='orange')
            plt.title("Cross-correlation for {}".format(var))
        if check_plot and peaks:
            plt.figure()
            plt.plot(merged.index, merged[str(var + "_y")].array, color='b')
            shifted_series = merged[str(var + "_x")].shift(lags[var])
            plt.plot(shifted_series.index, shifted_series.array, color='r')
    lag = round(sum(lags.values())/len(lags))
    lag_time = merged['times'][lag] - merged['times'][0]
    return lag_time



def find_lag_og(times, ts_x, ts_y, plot_cc=False, plot_lag=False):
    
    npts = len(times)
    
    warnings.filterwarnings("error")
    
    lags = []
    for x, y in zip(ts_x, ts_y):
        pccx = []
        pccy = []
        for j in range(npts-2): # prevent divide by zero due to std. dev. computation from diag(cov) of numpy's corrcoef Spearman function due to duplicates
            try:
                pccx.append(x.corr(y.shift(j), method='pearson'))
                pccy.append(y.corr(x.shift(j), method='pearson'))
            except RuntimeWarning:
                print(j, npts, x.name, y.name)
                # pass
        lag = []
        for pcc in [pccx, pccy]:
            peaks = find_peaks(pcc, height=0)
            try:
                best = np.array([(npts - peaks[0][p])*peaks[1].get('peak_heights')[p] for p in range(len(peaks[0]))])
                lag.append(peaks[0][best.argmax()])
            except ValueError:
                lag.append(np.array(pcc).argmax())
                peaks = False
        pcc = [pccx, pccy][np.array(lag).argmin()]
        lags.append(min(lag))
        
    
        if plot_cc:
            plt.figure()
            plt.plot(range(npts-2), pcc)
            plt.scatter(lags[-1], max(pcc), marker='x', color='orange')
            plt.title("Cross-correlation for {}".format(x.name))
            
        if plot_lag:
            plt.figure()
            plt.plot(range(npts), x.array, color='b')
            shifted_series = y.shift(lags[-1])
            plt.plot(range(npts), shifted_series.array, color='r')   
        
    warnings.resetwarnings() 
        
    lag = round(np.mean(lags))
    lag_time = t.Time(times[lag]) - t.Time(times[0])
            
    return lag, lag_time



# moved from timeseries
def get_lag(ts, resamp_factor=2, check_lag_plots=False):
    
    lags = []
    pcoefs = []
    ignore_duplicates = []
    
    for i in range(len(ts.data)):
        for j in range(len(ts.data)):
            if not (i == j) and not (j in ignore_duplicates):
                x = []
                for time in ts.times:
                    x.append(time.jd)
                spline1 = CubicSpline(x, ts.data[i])
                spline2 = CubicSpline(x, ts.data[j])
                # double sample points in timeseries for lag identification
                xs = np.arange(
                    x[0], 2*x[-1]-x[-2], (x[-1]-x[0])/(resamp_factor*len(x))
                    )
                c = np.correlate(spline1(xs)-np.mean(ts.data[i]), 
                                 spline2(xs)-np.mean(ts.data[j]), 
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
                    sc_index = {'bepi': 0, 'earth': 1, 'psp': 2, 'so': 3, 'sta': 4}
                    colors = ['indianred', 'darkgreen', 'slategrey', 
                              'steelblue', 'sandybrown', 'slategrey']
                    labels = ['BepiColombo', 'Earth', 'PSP', 
                              'Solar Orbiter', 'STEREO-A']
                    
                    # FIG 1: cross-correlation plot - max gives lag value
                    title = ('Cross-correlation of timeseries for {} and {}'
                             .format(labels[sc_index.get(ts.spacecraft[i])],
                                     labels[sc_index.get(ts.spacecraft[j])]))
                    
                    fig1 = plt.figure(figsize=(8,8), dpi=300)
                    ax = fig1.add_subplot()
                    xc = np.linspace(-len(c)/2, len(c)/2, len(c))
                    ax.plot(xc, c)
                    ax.set_title(title, pad=10, fontsize = 'x-large')
                    
                    # FIG 2: synchronised timeseries - features should overlap
                    title = 'Synchronized timeseries at {}'.format(
                        ', '.join([labels[sc_index.get(sc)] for sc in ts.spacecraft])
                        )
                    
                    fig2 = plt.figure(figsize=(8,8), dpi=300)
                    ax = fig2.add_subplot()
                    ax.set_ylabel('{} [{}]'.format(ts.variable, ts.units), 
                                  fontsize='large')
                    
                    xs_step_in_hours = t.TimeDelta(xs[1]-xs[0], format='jd').to_value('hr')
                    xs1 = np.arange(0, len(xs))
                    xs2 = xs1.copy() + lag
                    ax.plot(xs1*xs_step_in_hours, spline1(xs), 
                            color=colors[sc_index.get(ts.spacecraft[0])], 
                            label=labels[sc_index.get(ts.spacecraft[0])])
                    ax.plot(xs2*xs_step_in_hours, spline2(xs), 
                            color=colors[sc_index.get(ts.spacecraft[1])], 
                            label=labels[sc_index.get(ts.spacecraft[1])])
                    
                    ax.set_title(title, pad=45, fontsize = 'x-large')
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                              ncol=len(ts.spacecraft), frameon=False, fontsize='large')
                    ax.set_xlabel(r'$\mathrm{Duration \: [hours]}$', 
                                  fontsize='x-large', labelpad=10)

                lags.append(lag*ts.dt.to_value('hr'))
                pcoefs.append(np.corrcoef(synced_data1, synced_data2))
        ignore_duplicates.append(i)

    return lags, pcoefs

# moved from timeseries
def get_expected_lag(ts, conj, sim, coord, check_expected_plot=True):
    
    sim._get_times(mjd=True)
    
    lag = []
    for idx in range(len(conj.times)-1):
        
        V1, units, _ = sim.get_data('V1', conj.times[idx])
        V2, units, _ = sim.get_data('V2', conj.times[idx])
        V3, units, _ = sim.get_data('V3', conj.times[idx])
        
        r = []; theta = []; phi = []; sw_vel = []
        
        for i, coord in enumerate(conj.coords[idx]): 
            
            r.append(coord.distance.au)
            theta.append(coord.lat.rad)
            phi.append(coord.lon.rad)
            
            if sim.is_stationary:
                omega = (2*np.pi/25.38*u.rad/u.day).to(u.rad/u.hour)
                dt = t.TimeDelta(conj.times[idx] - sim.times[0]).to_value('hr')
                dlon = dt*omega.value
                phi_idx = sim._nearest(sim.phi.value + dlon, phi[i])
            else:
                phi_idx = sim._nearest(sim.phi.value, phi[i])
            theta_idx = sim._nearest(sim.theta.value, theta[i])
            r_idx = sim._nearest(sim.r.value, r[i])
            
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




def get_pearson_correlation(ts_x, ts_y, lag=0):
    
    pcc = []
    for x, y in zip(ts_x, ts_y):
        pcc.append(x.corr(y.shift(lag)))
        
    return np.mean(pcc)    


def merge_timeseries(coordinates, conj, trim_timeseries=0.1, add_features=False, attribute_to_conj=False):
    """
    Merges timeseries by resampling the one with the least number of data points.
    
    Parameters
    ----------
    coordinates: '~coordinates.Coordinates'
        Coordinates instance
    conj : '~conjunction.Conjunction'
        Conjunction instance from which timeseries are queried
    trim_timeseries: 'float'
        parameter with range = [0,1). If greater than 0, trims timeseries at each end by half that percentage.
        default value is 0.1, ie. 5% of all points are removed at each end.
    add_features: 'bool'
        if True, computes pre-defined features and appends them to the dataframe for each data point
    attribute_to_conj: 'bool'
        if True, output DataFrame becomes an attribute of conj object - accessed with conj.timeseries['merged']
    
    Returns
    -------
    '~pandas.DataFrame'
        DataFrame with times in the first column, 
        and values for each variable with _x and _y suffixes 
        referring to timeseries from one S/C or the other 
        in the subsequent columns.
    """
    if len(conj.sc_cell_times[conj.spacecraft[0]]) >= len(conj.sc_cell_times[conj.spacecraft[1]]):
        sc0 = conj.spacecraft[0]
        sc1 = conj.spacecraft[1]
    else:
        sc0 = conj.spacecraft[1]
        sc1 = conj.spacecraft[0]
        
    df0 = conj.timeseries[sc0]
    df1 = conj.timeseries[sc1]
        
    times0 = df0['times'].values
    times1 = df1['times'].values
    
    npts = len(times0)
    trim_npts = round(npts*trim_timeseries/2)
    trim = [trim_npts, npts-trim_npts]
    
    times = times0[trim[0]:trim[1]]
    
    df = {}
    df['times'] = times
    for column in df0.columns[1:]:
        spline = CubicSpline(times1, df1[column].values)
        df[column + '_x'] = df0[column].values[trim[0]:trim[1]]
        df[column + '_y'] = spline(times)
        
    coords0 = coordinates.get_sc_coordinates(spacecraft=sc0, times=times)
    coords1 = coordinates.get_sc_coordinates(spacecraft=sc1, times=times)

    df['lon_x'] = coords0.lon.rad
    df['lon_y'] = coords1.lon.rad
    df['lat_x'] = coords0.lat.rad
    df['lat_y'] = coords1.lat.rad
    df['dist_x'] = coords0.distance.au
    df['dist_y'] = coords1.distance.au   
    df['spacecraft_x'] = sc0
    df['spacecraft_y'] = sc1

    if conj.label[0] == None:
        df['labels'] = 'none'
    else:
        df['labels'] = conj.label[0]
            
    df = pd.DataFrame(df)
    
    if add_features:
        df = compute_features(conj, df)
    
    if attribute_to_conj:
        conj.timeseries['merged'] = df
        
    return df

def compute_features(conj, df):
    
    for var in conj.ts_units.keys():
        # compute gradients from splines
        splinex = CubicSpline(df['times'].values, df[var + '_x'])
        spliney = CubicSpline(df['times'].values, df[var + '_y'])
        df['d' + var + '_x'] = splinex(df['times'].values, 1)
        df['d' + var + '_y'] = spliney(df['times'].values, 1)
        
    # add difference of values
    for var in conj.ts_units.keys():
        df[var + '_diff'] = df[var + '_x'] - df[var + '_y']
    # add ratio of values
    for var in conj.ts_units.keys():
        df[var + '_ratio'] = df[var + '_x'] / df[var + '_y']
    
    # timeseries considered for lag determination
    # ts_x = [df['D_x'], df['T_x'], df['V1_x'], df['B1_x']]
    # ts_y = [df['D_y'], df['T_y'], df['V1_y'], df['B1_y']]
    ts_x = [df['V1_x']]
    ts_y = [df['V1_y']]
    
    lag, lag_time = find_lag(df['times'].values, ts_x, ts_y, conj, plot_cc=False, plot_lag=True)
    c1 = get_pearson_correlation(ts_x, ts_y)
    c2 = get_pearson_correlation(ts_x, ts_y, lag=lag)
    
    df['lag'] = t.TimeDelta(lag_time).to_value('hr')
    df['correlation'] = c1
    df['correlation_with_lag'] = c2
    
    # add kinetic energy and magnetic energy ratios
    df['E_k'] = ((df['D_x']*(df['V1_x']*df['V1_x'] + df['V2_x']*df['V2_x'] + df['V3_x']*df['V3_x'])) / (df['D_y']*(df['V1_y']*df['V1_y'] + df['V2_y']*df['V2_y'] + df['V3_y']*df['V3_y'])))
    df['E_mag'] = (df['B1_x']*df['B1_x'] + df['B2_x']*df['B2_x'] + df['B3_x']*df['B3_x']) / (df['B1_y']*df['B1_y'] + df['B2_y']*df['B2_y'] + df['B3_y']*df['B3_y'])
    
    return df

def reformat_timeseries(coordinates, conj_list, trim_timeseries=0.1, df=None, add_features=True, csv_filepath=None):
    """
    Parameters
    ----------
    coordinates: instantiated Coordinates class
    conj_list: list of conjunctions with timeseries
    csv_filepath : specify filepath if data should be saved to csv

    Returns
    -------
    df : pandas.DataFrame object containing timeseries and metadata.
    """
    coordinates._get_SPICE_kernels()
    
    if trim_timeseries < 0 or trim_timeseries >= 1:
        raise(Exception('trim_timeseries must be in the range [0, 1).'))
    
    for i, conj in enumerate(conj_list):
        if conj.times[-1] - conj.times[0] < 12*u.hour:
            pass
        else:
            try:
                df1 = merge_timeseries(coordinates, conj, trim_timeseries=trim_timeseries, add_features=add_features, attribute_to_conj=False)
                df = pd.concat([df, df1], ignore_index=True)
            except TypeError: # assign first conj data to df if df not provided, can't guarantee that conj at i==0 is not less than 12 hours
                df = merge_timeseries(coordinates, conj, trim_timeseries=trim_timeseries, add_features=add_features, attribute_to_conj=False)
            
    
    if csv_filepath:
        df.to_csv(csv_filepath, index=False)
    
    return df

def select_dataset(df, columns, label_column='labels', balance_dataset=True, csv_filepath='./Timeseries/example_subset.csv'):
    if balance_dataset:
        # find the indices corresponding to each label
        labels_ids = []
        for label in np.unique(df[label_column].values):
            labels_ids.append(np.where(df[label_column] == label)[0])
        # npts is the number of points for the label with the least data points
        npts = min([len(idx) for idx in labels_ids])
        ids = []
        for idx in labels_ids:
            np.random.shuffle(idx)
            ids.append(idx[:npts])
        ids = np.array(ids).flatten()
        df_subset = df.loc[ids, columns]
        labels = df.loc[ids, label_column]
    else:
        df_subset = df.loc[:, columns]
        labels = df.loc[:, label_column]
    
    print('The selected subset contains {} data points.'.format(np.shape(df_subset)[0]))
    
    if csv_filepath:
        df_subset.to_csv(csv_filepath, index=False)
        
    return df_subset, labels


##############################################################################
#                                    PCA                                     #
##############################################################################

def perform_PCA(df, y=None, no_of_components=3, scree=True, heatmap=True, plot_scatter=False):
    
    if not float(no_of_components).is_integer():
        raise Exception('no_of_components must be an integer.')
    no_of_components = int(no_of_components)
    if plot_scatter and no_of_components != 3:
        raise Exception('no_of_components must be 3 if plot_scatter is enabled.')
    
    
    ############    START OF PCA    ################

    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(df)
    idx = np.isnan(scaled_df).any(axis=1)
    scaled_df = scaled_df[~idx]
    if plot_scatter:
        y = y[~idx]
    try:
        pca = PCA(n_components=no_of_components)
        pca_0 = pca.fit_transform(scaled_df)
    except ValueError:
        n = min(df.shape)
        pca = PCA(n_components=n)
        pca_0 = pca.fit_transform(scaled_df)
        print("Invalid no_of_components given; no_of_components set to {}".format(n))

    #############    SCREE PLOT   ###############

    if scree:
        nums = np.arange(min(df.shape))
    
        var_ratio = []
        for num in nums:
            pca_test = PCA(n_components=num)
            pca_test.fit(scaled_df)
            var_ratio.append(np.sum(pca_test.explained_variance_ratio_))
    
        plt.figure(figsize=(6,4),dpi=250)
        plt.grid()
        plt.plot(nums,var_ratio,marker='o',c='steelblue')
        plt.xlabel('n_components')
        plt.ylabel('Explained variance ratio')
        plt.title('Cumulative Eigenvalues of Principal Components')

    ##########   HEATMAP of Feature contributions    ###########
    
    if heatmap:
        # Principal components correlation coefficients
        loadings = pca.components_
         
        # Number of features before PCA
        n_features = pca.n_features_in_
         
        # Feature names before PCA
        feature_names = df.columns.tolist()
         
        # PC names
        pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
         
        # Match PC names to loadings
        pc_loadings = dict(zip(pc_list, loadings))
         
        # Matrix of corr coefs between feature names and PCs
        loadings_df = pd.DataFrame.from_dict(pc_loadings)
        loadings_df['feature_names'] = feature_names
        loadings_df = loadings_df.set_index('feature_names')
    
        plt.figure(figsize=(10, 10), dpi=250)
    
        sns.heatmap(loadings_df, annot=False, cmap='coolwarm', center=0., linewidths=0.5)
    
        plt.title('Heatmap of SW properties\' contribution to PCs')
        plt.show()
        
    ####################   3D SCATTER PLOT   #####################
    
    if plot_scatter:
        
        Xax = pca_0[:,0]
        Yax = pca_0[:,1]
        Zax = pca_0[:,2]
        
        labels = {'cone':['indianred','X'], 
                  'quadrature':['darkgreen', '*'], 
                  'opposition':['sandybrown', 'd'], 
                  'parker spiral':['steelblue', 'o'],
                  'none':['slategrey','v']}
        
        fig = plt.figure(figsize=(8,10), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        
        fig.patch.set_facecolor('white')
        for l in list(labels.keys()):
            ix=np.where(y==l)
            ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=labels.get(l)[0], marker=labels.get(l)[1], s=5, label=l, alpha=0.5)
        
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.set_zlabel("PC 3", fontsize=12)
        
        ax.view_init(90,90)
        
        ax.legend()
        plt.show()
                
    return pca_0, y

##############################################################################
#                              DTW MATRIX PLOT                               #
##############################################################################

# 22 - 29 March solo/psp - 22/03 = 1944 - 29/03 = 2112
# 28 Aug - 11 Sep solo/sta - 28/08 = 5760 - 11/09 = 6096
# 29 Nov - 15 Dec solo/sta - 29/11 = 7992 - 15/12 = 8376

# get timeseries
# so_sub = so_v.iloc[1944:2112].values
# psp_sub = psp_v.iloc[1944:2112].values
# sta_sub = sta_v.iloc[7992:8376].values

def get_dtw_matrix(ts1, ts2, plot=True, times=None, 
                   ts_labels=['timeseries 1', 'timeseries 2']):
    """
    Calculates the dynamic time warping (DTW) similarity score, path and cost 
    matrix for two timeseries of the same length using tslearn. 
    Optionally plots the information.

    Parameters
    ----------
    ts1 : 'list or numpy array'
        Timeseries of length n. If sparse, missing values will be interpolated 
        using scipy.interpolate.CubicSpline
    ts2 : 'list or numpy array'
        Timeseries of length n. If sparse, missing values will be interpolated 
        using scipy.interpolate.CubicSpline
    plot : 'bool', optional
        Plot timeseries, cost matrix and optimal DTW path. 
        ts1 is plotted on the y-axis and ts2 is plotted on the x-axis.
        The default is True.
    ts_labels : 'list/tuple of strings', optional
        Labels for ts1 and ts2 respectively.

    Returns
    -------
    'float'
        DTW similarity score
    'list of integer pairs'
        Optimal DTW path with respect to distance cost

    """
    # TODO: need to scale timeseries (see behavior with density or B field ts)
    
    if any(np.isnan(y) for y in ts1):
        # interpolate nan values using simple linear interpolation 
        # to avoid issues at boundaries
        x1_data = []
        y1_data = []
        x1 = np.arange(len(ts1))
        for x, y in enumerate(ts1):
            if not np.isnan(y):
                x1_data.append(x)
                y1_data.append(y)
        y1 = np.interp(x1, x1_data, y1_data)
    else:
        x1_data = False; y1_data = False
        x1 = np.arange(len(ts1))
        y1 = ts1
    
    if any(np.isnan(y) for y in ts2):
        x2_data = []
        y2_data = []
        x2 = np.arange(len(ts2))
        for x, y in enumerate(ts2):
            if not np.isnan(y):
                x2_data.append(x)
                y2_data.append(y)
        y2 = np.interp(x2, x2_data, y2_data)
    else:
        x2_data = False; y2_data = False
        x2 = np.arange(len(ts2))
        y2 = ts2
    
    # perform dtw
    path, sim = metrics.dtw_path(y1, y2)
    
    # compute distance matrix
    mat = cdist(np.array([x1, y1]).transpose(), np.array([x2, y2]).transpose())
    
    # make figure
    plt.figure(1, figsize=(8, 8))
    
    sz = x1.shape[0]
    
    # define axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02
    
    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]
    
    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)
    
    # plot matrix
    ax_gram.imshow(mat, origin='lower')
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    # plot DTW path
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], color='r', linewidth=2)
    
    # plot ts1 on y-axis
    ax_s_y.plot(- y1, x1, 'k', linewidth=2)
    if x1_data:
        interp_regions = set(x1) - set(x1_data)
        for r in interp_regions:
            ax_s_y.axhspan(r-0.4, r+0.4, color='grey', alpha=0.2)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz - 1))
    
    # plot ts2 on x-axis
    ax_s_x.plot(x2, y2, 'k', linewidth=2)
    if x2_data:
        interp_regions = set(x2) - set(x2_data)
        for r in interp_regions:
            ax_s_x.axvspan(r-0.4, r+0.4, color='grey', alpha=0.2)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz - 1))
    
    plt.figtext(0.5, 0.97, ts_labels[1], fontsize=14)
    plt.figtext(-0.02, 0.4, ts_labels[0], fontsize=14, rotation=90)
    if times:
        plt.figtext(0, 0.82, ' {} \n to \n {}'.format(times[0], times[-1]), fontsize=14)
    
    # plt.tight_layout()
    plt.show()
    
    return sim, path


# TODO: implement scaling & trimming function for timeseries
# TODO: implement approx. scaling & lag based on s/c trajectory