import os
import sys

import automatic_detection as autodet
root = autodet.cfg.base

sys.path.append(os.path.join(root, 'maps'))
import map_utils
from definition_subregions import *

sys.path.append(os.path.join(root, 'focal_mechanisms/'))
import utils_focmech
sys.path.append(os.path.join(root, 'focal_mechanisms/FMC/'))
import plotFMC
from functionsFMC import kave, mecclass
sys.path.append(os.path.join(root,'focal_mechanisms/stress_inversion/'))
import utils_stress

import numpy as np
import h5py as h5
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time as give_time

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colorbar as clb
import matplotlib.gridspec as gridspec

import mplstereonet
import cartopy as ctp

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

from obspy import UTCDateTime as udt
from obspy.geodetics.base import calc_vincenty_inverse
from obspy.imaging.beachball import beach

import seaborn as sns
sns.set()
sns.set_style('ticks')
#sns.set_context('paper')
sns.set_palette('colorblind')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['svg.fonttype'] = 'none'

_colors_ = ['C0', 'C1']

data_path = '/home/eric/Dropbox (MIT)/stress_tensor_inversion'
data = pd.read_csv(os.path.join(data_path, 'central_Crete_formatted.csv'))
# try to exclude the planes that were tricky to convert to fm
#data.drop(index=[11,19,20,21,23,24], inplace=True)

strikes_1 = data['strike'].values
dips_1 = data['dip'].values
rakes_1 = np.float32(data['rake'].values)%360.
n_earthquakes = len(strikes_1)
planes_2 = [utils_focmech.aux_plane(s, d, r) for (s, d, r)
            in zip(strikes_1, dips_1, rakes_1)]
strikes_2, dips_2, rakes_2 = np.float32([planes_2[i][0] for i in range(n_earthquakes)]),\
                             np.float32([planes_2[i][1] for i in range(n_earthquakes)]),\
                             np.float32([planes_2[i][2] for i in range(n_earthquakes)])

def stress_inversion(strikes_1, dips_1, rakes_1):
    n_earthquakes = len(strikes_1)
    planes_2 = [utils_focmech.aux_plane(s, d, r) for (s, d, r)
                in zip(strikes_1, dips_1, rakes_1)]
    strikes_2, dips_2, rakes_2 = np.float32([planes_2[i][0] for i in range(n_earthquakes)]),\
                                 np.float32([planes_2[i][1] for i in range(n_earthquakes)]),\
                                 np.float32([planes_2[i][2] for i in range(n_earthquakes)])
    # --------------------------------
    #    stress tensor inversion
    # --------------------------------
    # inversion parameter
    friction_min = 0.2
    friction_max = 0.8
    friction_step = 0.05
    n_random_selections = 30
    n_stress_iter = 20
    n_averaging = 5
    n_bootstraps = 1000
    Michael_kwargs = {}
    Michael_kwargs['max_n_iterations'] = 1000
    Michael_kwargs['shear_update_atol'] = 1.e-7
    Tarantola_kwargs0 = {}
    inversion_output = {}
    for experiment in ['true_fp_linear', 'true_fp_iterative']:
        inversion_output[experiment] = {}
        inversion_output[experiment]['stress_tensor'] =\
                np.zeros((3, 3), dtype=np.float32)
        inversion_output[experiment]['principal_stresses'] =\
                np.zeros(3, dtype=np.float32)
        inversion_output[experiment]['principal_directions'] =\
                np.zeros((3, 3), dtype=np.float32)
        inversion_output[experiment]['misfit'] = 0.
        inversion_output[experiment]['boot_stress_tensor'] =\
                np.zeros((n_bootstraps, 3, 3), dtype=np.float32)
        inversion_output[experiment]['boot_principal_stresses'] =\
                np.zeros((n_bootstraps, 3), dtype=np.float32)
        inversion_output[experiment]['boot_principal_directions'] =\
                np.zeros((n_bootstraps, 3, 3), dtype=np.float32)
        inversion_output[experiment]['boot_misfit'] = np.zeros(n_bootstraps, dtype=np.float32)
    # ----------- whole data set
    inversion_output['true_fp_linear']['stress_tensor'],\
    inversion_output['true_fp_linear']['principal_stresses'],\
    inversion_output['true_fp_linear']['principal_directions'] =\
               utils_stress.Michael1984_inversion(
                       strikes_1, dips_1, rakes_1, return_eigen=True, return_stats=False,
                       Tarantola_kwargs=Tarantola_kwargs0)
    inversion_output['true_fp_linear']['misfit'] =\
            np.mean(utils_stress.mean_angular_residual(
                inversion_output['true_fp_linear']['stress_tensor'],
                strikes_1, dips_1, rakes_1))
    inversion_output['true_fp_iterative']['stress_tensor'], _,\
    inversion_output['true_fp_iterative']['principal_stresses'],\
    inversion_output['true_fp_iterative']['principal_directions'] =\
               utils_stress.Michael1984_iterative(
                       strikes_1, dips_1, rakes_1, return_eigen=True, return_stats=False,
                       Tarantola_kwargs=Tarantola_kwargs0, **Michael_kwargs)
    inversion_output['true_fp_iterative']['misfit'] =\
            np.mean(utils_stress.mean_angular_residual(
                inversion_output['true_fp_iterative']['stress_tensor'],
                strikes_1, dips_1, rakes_1))
    # ----------- bootstrapped data set
    for b in range(n_bootstraps):
        bootstrapped_set = np.random.choice(
                np.arange(n_earthquakes), replace=True, size=n_earthquakes)
        strikes_b, dips_b, rakes_b = strikes_1[bootstrapped_set], dips_1[bootstrapped_set], rakes_1[bootstrapped_set]
        inversion_output['true_fp_linear']['boot_stress_tensor'][b, ...],\
        inversion_output['true_fp_linear']['boot_principal_stresses'][b, ...],\
        inversion_output['true_fp_linear']['boot_principal_directions'][b, ...] =\
                   utils_stress.Michael1984_inversion(
                           strikes_b, dips_b, rakes_b, return_eigen=True, return_stats=False,
                           Tarantola_kwargs=Tarantola_kwargs0)
        inversion_output['true_fp_linear']['boot_misfit'][b] =\
                np.mean(utils_stress.mean_angular_residual(
                    inversion_output['true_fp_linear']['boot_stress_tensor'][b, ...],
                    strikes_b, dips_b, rakes_b))
        inversion_output['true_fp_iterative']['boot_stress_tensor'][b, ...], _,\
        inversion_output['true_fp_iterative']['boot_principal_stresses'][b, ...],\
        inversion_output['true_fp_iterative']['boot_principal_directions'][b, ...] =\
                   utils_stress.Michael1984_iterative(
                           strikes_b, dips_b, rakes_b, return_eigen=True, return_stats=False,
                           Tarantola_kwargs=Tarantola_kwargs0, **Michael_kwargs)
        inversion_output['true_fp_iterative']['boot_misfit'][b] =\
                np.mean(utils_stress.mean_angular_residual(
                    inversion_output['true_fp_iterative']['boot_stress_tensor'][b, ...],
                    strikes_b, dips_b, rakes_b))
    inversion_output['strikes'] = np.stack((strikes_1, strikes_2), axis=1)
    inversion_output['dips'] = np.stack((dips_1, dips_2), axis=1)
    inversion_output['rakes'] = np.stack((rakes_1, rakes_2), axis=1)
    return inversion_output

def hist2d(azimuths, plunges, nbins=200, smoothing_sig=0, plot=False):
    lons, lats = mplstereonet.stereonet_math.line(plunges, azimuths)
    count, lon_bins, lat_bins = np.histogram2d(
            lons, lats, range=([-np.pi/2., np.pi/2.], [-np.pi/2., np.pi/2.]), bins=nbins)
    lons_g, lats_g = np.meshgrid((lon_bins[1:] + lon_bins[:-1])/2.,
                                 (lat_bins[1:] + lat_bins[:-1])/2.,
                                 indexing='ij')
    if smoothing_sig > 0:
        count = gaussian_filter(count, smoothing_sig)
    if plot:
        fig = plt.figure('2d_histogram_stereo', figsize=(18, 9))
        ax = fig.add_subplot(111, projection='stereonet')
        pcl = ax.pcolormesh(lons_g, lats_g, count)
        plt.colorbar(mappable=pcl)
    return count, lons_g, lats_g

def get_CI_levels(azimuths, plunges, confidence_intervals=[95., 90.],
                  nbins=200, smoothing_sig=1, return_count=False,
                  plot=False):
    print(nbins, smoothing_sig, return_count, confidence_intervals)
    # get histogram on a 2d grid
    count, lons_g, lats_g = hist2d(azimuths, plunges, nbins=nbins, smoothing_sig=smoothing_sig)
    # flatten the count array and sort it from largest to smallest
    count_vector = np.sort(count.copy().flatten())[::-1]
    # compute the "CDF" of the counts
    count_CDF = np.hstack(([0.], np.cumsum(count_vector)/count_vector.sum(), [1.]))
    count_vector = np.hstack(([count_vector.max()+1.], count_vector, [0.]))
    # build an interpolator that gives the count number 
    # for a given % of the total mass
    mass_dist = interp1d(100.*count_CDF, count_vector)
    mass_dist_ = lambda x: mass_dist(x).item()
    #confidence_intervals = list(map(mass_dist_, [(100-k) for k in confidence_intervals]))
    confidence_intervals = list(map(mass_dist_, [k for k in confidence_intervals]))
    if plot:
        fig = plt.figure('2d_histogram_stereo', figsize=(18, 9))
        ax = fig.add_subplot(111, projection='stereonet')
        pcl = ax.pcolormesh(lons_g, lats_g, count)
        ax.contour(lons_g, lats_g, count, levels=confidence_intervals, zorder=2, cmap='jet')
        plt.colorbar(mappable=pcl)
    if return_count:
        return count, lons_g, lats_g, confidence_intervals
    else:
        return confidence_intervals

def plot_inverted_stress_tensors(inversion_output, axes=None, **kwargs):
    hist_kwargs = {}
    hist_kwargs['smoothing_sig'] = kwargs.get('smoothing_sig', 1)
    hist_kwargs['nbins'] = kwargs.get('nbins', 200)
    hist_kwargs['return_count'] = kwargs.get('return_count', True)
    hist_kwargs['confidence_intervals'] = kwargs.get('confidence_intervals', [95.])
    markers = ['o', 's', 'v']
    methods = ['linear', 'iterative']
    n_bootstraps = inversion_output['true_fp_linear']['boot_principal_directions'].shape[0]
    fig = plt.figure('inverted_stress_tensors_Crete', figsize=(18, 9))
    fig.suptitle('Central Crete')
    gs1 = fig.add_gridspec(nrows=3, ncols=3, top=0.88, bottom=0.11,
                           left=0.05, right=0.95, hspace=0.4, wspace=0.4)
    gs2 = fig.add_gridspec(nrows=3, ncols=3, top=0.88, bottom=0.11,
                           left=0.20, right=0.80, hspace=0.4, wspace=0.7)
    axes = []
    ax1 = fig.add_subplot(gs2[0, 1], projection='stereonet')
    ax2 = fig.add_subplot(gs2[0, 2])
    ax3 = fig.add_subplot(gs2[0, 0], projection='stereonet')
    ax3.set_title('Fault planes', pad=30)
    ax3.plane(inversion_output['strikes'][:, 0],
              inversion_output['dips'][:, 0], color='k', lw=1.0)
    for j, method in enumerate(methods):
        exp = f'true_fp_{method}'
        R = utils_stress.R_(inversion_output[exp]['principal_stresses'])
        for k in range(3):
            if k == 0:
                label = '{}:\n'.format(method.replace('_', ' ').capitalize())
                for k2 in range(3):
                    az, pl = utils_focmech.get_bearing_plunge(
                            inversion_output[exp]['principal_directions'][:, k2])
                    label += r'$\sigma_{{{:d}}}$: {:.1f}'u'\u00b0''|{:.1f}'u'\u00b0'', '.\
                            format(k2+1, az, pl)
                label += ' R={:.2f}, $\\vert{{\\Delta \\theta}}\\vert$={:.1f}'u'\u00b0'.\
                        format(R, inversion_output[exp]['misfit'])
            else:
                label = ''
            az, pl = utils_focmech.get_bearing_plunge(
                    inversion_output[exp]['principal_directions'][:, k])
            ax1.line(pl, az, marker=markers[k], markeredgecolor=_colors_[j], markeredgewidth=2,
                     markerfacecolor='none', markersize=[15, 15][j], label=label, zorder=2)
            boot_pd_stereo = np.zeros((n_bootstraps, 2), dtype=np.float32)
            for b in range(n_bootstraps):
                boot_pd_stereo[b, :] = utils_focmech.get_bearing_plunge(
                        inversion_output[exp]['boot_principal_directions'][b, :, k])
            count, lons_g, lats_g, levels = get_CI_levels(
                    boot_pd_stereo[:, 0], boot_pd_stereo[:, 1], **hist_kwargs)
            ax1.contour(lons_g, lats_g, count, levels=levels, vmin=0.,
                        linestyles=['solid', 'dashed', 'dashdot'][k],
                        linewidths=0.75, colors=_colors_[j])
        axes.append(ax1)
        Rs = np.zeros(n_bootstraps, dtype=np.float32)
        for b in range(n_bootstraps):
            Rs[b] = utils_stress.R_(inversion_output[exp]['boot_principal_stresses'][b, :])
        ax2.hist(Rs, range=(0., 1.), bins=20, lw=2.5, color=_colors_[j], histtype='step')
    ax2.set_xlabel('Shape Ratio')
    ax2.set_ylabel('Count')
    for ax in axes:
        #ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(-0.5, -0.30))
    plt.subplots_adjust(top=0.88, bottom=0.11,
                        left=0.05, right=0.95,
                        hspace=0.4, wspace=0.4)
    return fig


def FMSI_fm_coordinates(strikes_1, dips_1, rakes_1,
                        strikes_2, dips_2, rakes_2):
    n1, _ = utils_stress.normal_slip_vectors(
            strikes_1, dips_1, rakes_1)
    n2, _ = utils_stress.normal_slip_vectors(
            strikes_2, dips_2, rakes_2)
    # define right-handed coordinate system (n1, b, n2):
    n1, n2 = n1.T, n2.T
    b = np.cross(n2, n1)
    return n1, b, n2

def FMSI_rotation_matrix(principal_directions, strikes_1, dips_1, rakes_1):
    n_earthquakes = len(strikes_1)
    # compute auxiliary planes
    planes_2 = [utils_focmech.aux_plane(s, d, r) for (s, d, r)
                in zip(strikes_1, dips_1, rakes_1)]
    strikes_2, dips_2, rakes_2 = np.float32([planes_2[i][0] for i in range(n_earthquakes)]),\
                                 np.float32([planes_2[i][1] for i in range(n_earthquakes)]),\
                                 np.float32([planes_2[i][2] for i in range(n_earthquakes)])
    # compute focal mechanisms' coordinate systems (X')
    Xp = np.stack(FMSI_fm_coordinates(strikes_1, dips_1, rakes_1,
                                      strikes_2, dips_2, rakes_2),
                  axis=2)
    X = principal_directions
    # build the angle cosine matrix
    beta = np.zeros((n_earthquakes, 3, 3), dtype=np.float32)
    for n in range(n_earthquakes):
        for i in range(3):
            for j in range(3):
                beta[n, i, j] = np.sum(Xp[n, :, i]*X[:, j])
    return beta

def FMSI_rotation_angle(principal_directions, R,
                        strikes_1, dips_1, rakes_1):
    
    Phi = 1. - R
    beta = FMSI_rotation_matrix(
            principal_directions, strikes_1, dips_1, rakes_1)
    rot1 = np.arctan((beta[:, 0, 2]*beta[:, 1, 2] + Phi*beta[:, 0, 1]*beta[:, 1, 1])
                    /(Phi*beta[:, 0, 1]*beta[:, 2, 1] + beta[:, 0, 2]*beta[:, 2, 2]))
    rot2 = np.arctan((beta[:, 0, 2]*beta[:, 1, 2] + Phi*beta[:, 0, 1]*beta[:, 1, 1])
                    /(Phi*beta[:, 1, 1]*beta[:, 2, 1] + beta[:, 1, 2]*beta[:, 2, 2]))
    k = (Phi*beta[:, 0, 1]**2 + beta[:, 0, 2]**2 - Phi*beta[:, 1, 1]**2 - beta[:, 1, 2]**2)\
            /(Phi*beta[:, 0, 1]*beta[:, 1, 1] + beta[:, 0, 2]*beta[:, 1, 2])
    rot3_plus = -np.arccos(np.sqrt(0.5 + np.sqrt(0.25 - 1./(4. + k**2))))*np.sign(k)
    rot3_minus = -np.arccos(np.sqrt(0.5 - np.sqrt(0.25 - 1./(4. + k**2))))*np.sign(k)
    r2d = 180./np.pi
    return rot1*r2d, rot2*r2d, rot3_plus*r2d, rot3_minus*r2d

def test_FMSI(inversion_output,
              strikes_1, dips_1, rakes_1,
              strikes_2, dips_2, rakes_2):
    """
    Test my FMSI routines by checking if I retrieve a value of R
    close to the one from my inversion.
    """
    for method in inversion_output.keys():
        pd = inversion_output[method]['principal_directions']
        R = utils_stress.R_(inversion_output[method]['principal_stresses'])
        # when calculating the rotation matrix, we don't know which
        # planes are the fault planes, so we need to compute one matrix
        # for each set of planes
        beta1 = FMSI_rotation_matrix(pd, strikes_1, dips_1, rakes_1)
        beta2 = FMSI_rotation_matrix(pd, strikes_2, dips_2, rakes_2)
        Phi1 = -(beta1[:, 0, 2]*beta1[:, 1, 2])/(beta1[:, 0, 1]*beta1[:, 1, 1])
        Phi2 = -(beta2[:, 0, 2]*beta2[:, 1, 2])/(beta2[:, 0, 1]*beta2[:, 1, 1])
        Rs = []
        for i in range(len(strikes_1)):
            if 0. < Phi1[i] < 1.0:
                Rs.append(1.-Phi1[i])
            if 0. < Phi2[i] < 1.0:
                Rs.append(1.-Phi2[i])
        R_FMSI = np.float32(Rs).mean()
        print('{}: my R={:.2f}, FMSI R={:.2f}'.format(
            method.replace('_', ' ').capitalize(), R, R_FMSI))

def test_Poyraz_results():
    # express principal directions in the (north, west, up) coordinate system
    #sig1_az, sig1_pl = 110., 0.
    #sig2_az, sig2_pl = 201., 58.
    #sig3_az, sig3_pl = 20., 32.
    sig1_az, sig1_pl = 103, 27.
    sig2_az, sig2_pl = 256., 61.
    sig3_az, sig3_pl = 7., 11.
    d2r = np.pi/180.
    u1_n = np.cos(sig1_az*d2r)*np.cos(sig1_pl*d2r)
    u1_w = -np.sin(sig1_az*d2r)*np.cos(sig1_pl*d2r)
    u1_z = np.sin(sig1_pl*d2r)
    u2_n = np.cos(sig2_az*d2r)*np.cos(sig2_pl*d2r)
    u2_w = -np.sin(sig2_az*d2r)*np.cos(sig2_pl*d2r)
    u2_z = np.sin(sig2_pl*d2r)
    u3_n = np.cos(sig3_az*d2r)*np.cos(sig3_pl*d2r)
    u3_w = -np.sin(sig3_az*d2r)*np.cos(sig3_pl*d2r)
    u3_z = np.sin(sig3_pl*d2r)
    sig1 = np.array([u1_n, u1_w, u1_z])
    sig2 = np.array([u2_n, u2_w, u2_z])
    sig3 = -np.array([u3_n, u3_w, u3_z])
    R = 0.45
    V = np.stack((sig1, sig2, sig3), axis=1)
    s1, s2, s3 = -1., 2*R-1., +1.
    S = np.diag(np.array([s1, s2, s3]) - np.sum([s1, s2, s3])/3.)
    S /= np.sqrt(np.sum(S**2))
    stress_tensor = np.dot(V, np.dot(S, V.T))
    # ------------------------------------
    #angles_1 = utils_stress.angular_residual(
    #        stress_tensor, strikes_1, dips_1, rakes_1)
    #angles_2 = utils_stress.angular_residual(
    #        stress_tensor, strikes_2, dips_2, rakes_2)
    # -------------------------------------
    ps_P, pd_P = utils_stress.stress_tensor_eigendecomposition(stress_tensor)
    rot11, rot21, rot3_p1, rot3_m1 = FMSI_rotation_angle(pd_P, 0.45, strikes_1, dips_1, rakes_1)
    rot12, rot22, rot3_p2, rot3_m2 = FMSI_rotation_angle(pd_P, 0.45, strikes_2, dips_2, rakes_2)
    min_rot1 = np.min(np.abs(np.stack((rot11, rot21, rot3_p1, rot3_m1), axis=1)), axis=1)
    min_rot2 = np.min(np.abs(np.stack((rot12, rot22, rot3_p2, rot3_m2), axis=1)), axis=1)
    strikes_FMSI = np.zeros(len(strikes_1))
    dips_FMSI = np.zeros(len(strikes_1))
    rakes_FMSI = np.zeros(len(strikes_1))
    strikes_FMSI[min_rot1 < min_rot2] = strikes_1[min_rot1 < min_rot2]
    strikes_FMSI[min_rot1 > min_rot2] = strikes_2[min_rot1 > min_rot2]
    dips_FMSI[min_rot1 < min_rot2] = dips_1[min_rot1 < min_rot2]
    dips_FMSI[min_rot1 > min_rot2] = dips_2[min_rot1 > min_rot2]
    rakes_FMSI[min_rot1 < min_rot2] = rakes_1[min_rot1 < min_rot2]
    rakes_FMSI[min_rot1 > min_rot2] = rakes_2[min_rot1 > min_rot2]
    I, fp_strikes, fp_dips, fp_rakes = utils_stress.compute_instability_parameter(
            pd_P, R, 0.6, strikes_1, dips_1, rakes_1, strikes_2, dips_2, rakes_2,
            return_fault_planes=True)
    # plot the fault planes selected with the different methods
    fig = plt.figure('selected_fault_planes', figsize=(18, 9))
    ax1 = fig.add_subplot(2, 2, 1, projection='stereonet')
    ax1.set_title('Instability criterion', pad=30.)
    ax1.plane(fp_strikes, fp_dips, color='k')
    ax2 = fig.add_subplot(2, 2, 2, projection='stereonet')
    ax2.set_title('Min. rotation criterion', pad=30.)
    ax2.plane(strikes_FMSI, dips_FMSI, color='k')




inversion_output = stress_inversion(strikes_1, dips_1, rakes_1)

fig = plot_inverted_stress_tensors(inversion_output, smoothing_sig=4)
