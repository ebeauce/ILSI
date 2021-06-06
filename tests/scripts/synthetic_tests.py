import os
import sys

sys.path.append(os.path.join(root,'focal_mechanisms/stress_inversion/'))
import utils_stress

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from time import time as give_time

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colorbar as clb
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mplstereonet

import seaborn as sns
sns.set()
sns.set_style('ticks')
sns.set_palette('colorblind')
#sns.set_context('paper')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['svg.fonttype'] = 'none'

#_colors_ = ['C0', 'C1', 'C2', 'C4']
_colors_ = ['C0', 'C4', 'C1', 'C2']

def main(experiment=1):
    # --------------------------------------
    #   synthetic experiment parameters
    # --------------------------------------
    # fault plane parameters
    n_earthquakes = 100
    np.random.seed(0)
    rakes = []
    if experiment == 1:
        # -------- experiment #1 ------------
        strike_mean, strike_std = 110., 10.
        dip_min, dip_max = 65., 90.
        strikes = np.random.normal(loc=strike_mean, scale=strike_std, size=n_earthquakes)
        dips = np.random.uniform(low=dip_min, high=dip_max, size=n_earthquakes)
        # stress tensor in the (north, west, upward) coordinate system
        sig1 = np.array([-1./np.sqrt(2.), -1./np.sqrt(2.), 0.])
        sig2 = np.array([0., 0., +1.])
        sig3 = np.array([-1./np.sqrt(2.), 1./np.sqrt(2.), 0.])
        R = 0.50
    elif experiment == 2:
        # -------- experiment #2 ------------
        strike_mean, strike_std = 110., 10.
        dip_min, dip_max = 20., 65.
        strikes = np.random.normal(loc=strike_mean, scale=strike_std, size=n_earthquakes)
        dips = np.random.uniform(low=dip_min, high=dip_max, size=n_earthquakes)
        # stress tensor in the (north, west, upward) coordinate system
        sig1 = np.array([-1./np.sqrt(2.), -1./np.sqrt(2.), 0.])
        sig2 = np.array([0., 0., +1.])
        sig3 = np.array([-1./np.sqrt(2.), 1./np.sqrt(2.), 0.])
        V = np.stack((sig1, sig2, sig3), axis=1)
        # do the rotation in the eigenbasis:
        # rotate 45 degrees around sig3
        unit_vectors = np.identity(3)
        rot = rotation_x3(45.)
        unit_vectors = np.dot(rot, unit_vectors)
        # take these vectors back to the original frame (north, west, upward)
        new_principal_directions = np.dot(unit_vectors, V.T)
        sig1 = new_principal_directions[:, 0]
        sig2 = new_principal_directions[:, 1]
        sig3 = new_principal_directions[:, 2]
        R = 0.70
    # -------------------
    # tension positive
    V = np.stack((sig1, sig2, sig3), axis=1)
    s1, s2, s3 = -1., 2*R-1., +1.
    S = np.diag(np.array([s1, s2, s3]) - np.sum([s1, s2, s3])/3.)
    S /= np.sqrt(np.sum(S**2))
    true_stress_tensor = np.dot(V, np.dot(S, V.T))
    # eigenvalue decomposition
    true_princ_stresses, true_princ_dir = \
            utils_stress.stress_tensor_eigendecomposition(true_stress_tensor)
    print('True stress tensor:\n', true_stress_tensor)
    print('The directions of the principal stresses are:')
    for i, label in enumerate(['Most compressive', 'Intermediate', 'Least compressive']):
        print('{} stress: Azimuth={:.2f}, plunge={:.2f}'.
                format(label, utils_focmech.get_bearing_plunge(true_princ_dir[:, i])[0],
                       utils_focmech.get_bearing_plunge(true_princ_dir[:, i])[1]))
    #mt = pmt.MomentTensor(mnn=true_stress_tensor[0,0], mee=true_stress_tensor[1,1],
    #                      mdd=true_stress_tensor[2,2], mne=-true_stress_tensor[0,1],
    #                      mnd=-true_stress_tensor[0,2], med=true_stress_tensor[1,2])
    #[s1, d1, r1], [s2, d2, r2] = mt.both_strike_dip_rake()
    #strikes = np.random.normal(loc=s1, scale=15., size=n_earthquakes)%360.
    #dips = np.random.normal(loc=d1, scale=20., size=n_earthquakes)
    #dips = np.clip(dips, 0., 90.)
    # determine rakes from strikes/dips and stress tensor
    for i in range(len(strikes)):
        # give fake rake, we only want the normal
        n, _ = utils_focmech.normal_slip_vectors(strikes[i], dips[i], 0.)
        traction = np.dot(true_stress_tensor, n.T) 
        normal_traction = np.sum(traction.squeeze()*n.squeeze(), axis=-1)*n.T
        shear_traction = traction - normal_traction
        shear_dir = shear_traction/np.sqrt(np.sum(shear_traction**2))
        # find the rake that will make slip in the same direction as shear
        s, d, r = utils_focmech.strike_dip_rake(n, shear_dir)
        rakes.append(r)
    rakes = np.float32(rakes)%360.
    # generate noisy focal mechanisms
    noise = [0., 3., 10.]
    strikes_1 = np.zeros((n_earthquakes, len(noise)), dtype=np.float32)
    dips_1 = np.zeros((n_earthquakes, len(noise)), dtype=np.float32)
    rakes_1 = np.zeros((n_earthquakes, len(noise)), dtype=np.float32)
    strikes_2 = np.zeros((n_earthquakes, len(noise)), dtype=np.float32)
    dips_2 = np.zeros((n_earthquakes, len(noise)), dtype=np.float32)
    rakes_2 = np.zeros((n_earthquakes, len(noise)), dtype=np.float32)
    for i in range(n_earthquakes):
        for n in range(len(noise)):
            s1 = strikes[i] + np.random.uniform(low=-noise[n], high=noise[n])
            d1 = np.random.uniform(low=max(0., dips[i]-noise[n]), high=min(90., dips[i]+noise[n]))
            r1 = rakes[i] + np.random.uniform(low=-noise[n], high=noise[n])
            strikes_1[i, n], dips_1[i, n], rakes_1[i, n] =\
                    s1, d1, r1
            s2, d2, r2 = utils_focmech.aux_plane(
                    strikes_1[i, n], dips_1[i, n], rakes_1[i, n])
            strikes_2[i, n], dips_2[i, n], rakes_2[i, n] =\
                    s2, d2, r2
    # make sure the parameters fall into the correct range
    strikes_1, rakes_1 = strikes_1%360., rakes_1%360. 
    strikes_2, rakes_2 = strikes_2%360., rakes_2%360.
    print('-------------')
    print(np.allclose(strikes, strikes_1[:, 0]))
    print(np.allclose(dips, dips_1[:, 0]))
    print(np.allclose(rakes, rakes_1[:, 0]))
    # --------------------------------------
    #  shear tractions on the planes defined by
    #  the noisy focal mechanisms
    # --------------------------------------
    shear_tractions = np.zeros((n_earthquakes, len(noise), 3), dtype=np.float32)
    slip_vectors = np.zeros((len(noise), n_earthquakes, 3), dtype=np.float32)
    for n in range(len(noise)):
        for i in range(n_earthquakes):
            n_, d_ = utils_stress.normal_slip_vectors(
                    strikes_1[i, n], dips_1[i, n], rakes_1[i, n])
            _, _, shear_tractions[i, n, :] = utils_stress.compute_traction(
                    true_stress_tensor, n_.reshape(1, 3))
            slip_vectors[n, i, :] = d_
    print('Normalized shear traction on true faults using the true stress tensor:')
    print(np.sqrt(np.sum(shear_tractions[:, 0, :]**2, axis=-1)))
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
    #Tarantola_kwargs0['C_d_inv'] = 0.1*np.diag(np.ones(n_earthquakes*3, dtype=np.float32))
    #Tarantola_kwargs0['C_d_inv'] = np.diag(np.random.uniform(low=5., high=100., size=n_earthquakes*3))
    #Tarantola_kwargs0['C_d'] = np.diag(1./np.diag(Tarantola_kwargs0['C_d_inv']))
    #Tarantola_kwargs0['C_m_inv'] = 0.2*np.diag(np.ones(5, dtype=np.float32))
    #Tarantola_kwargs0['C_m'] = np.diag(1./np.diag(Tarantola_kwargs0['C_m_inv']))
    #Tarantola_kwargs['inversion_space'] = 'data_space'
    inversion_output = {}
    methods = ['linear', 'failure_criterion', 'iterative',
               'iterative_failure_criterion']
    for method in methods:
        inversion_output[method] = {}
        inversion_output[method]['stress_tensor'] =\
                np.zeros((len(noise), 3, 3), dtype=np.float32)
        inversion_output[method]['principal_stresses'] =\
                np.zeros((len(noise), 3), dtype=np.float32)
        inversion_output[method]['principal_directions'] =\
                np.zeros((len(noise), 3, 3), dtype=np.float32)
        inversion_output[method]['misfit'] = np.zeros(len(noise), dtype=np.float32)
        inversion_output[method]['boot_stress_tensor'] =\
                np.zeros((len(noise), n_bootstraps, 3, 3), dtype=np.float32)
        inversion_output[method]['boot_principal_stresses'] =\
                np.zeros((len(noise), n_bootstraps, 3), dtype=np.float32)
        inversion_output[method]['boot_principal_directions'] =\
                np.zeros((len(noise), n_bootstraps, 3, 3), dtype=np.float32)
        inversion_output[method]['boot_misfit'] = np.zeros((len(noise), n_bootstraps), dtype=np.float32)
    inversion_output['iterative_failure_criterion']['friction'] =\
            np.zeros(len(noise), dtype=np.float32)
    inversion_output['failure_criterion']['friction'] =\
            np.zeros(len(noise), dtype=np.float32)
    #inversion_output['iterative_failure_criterion']['boot_friction'] =\
    #        np.zeros((len(noise), n_bootstraps), dtype=np.float32)
    #inversion_output['failure_criterion']['boot_friction'] =\
    #        np.zeros((len(noise), n_bootstraps), dtype=np.float32)
    for experiment in ['true_fp_linear', 'true_fp_iterative']:
        inversion_output[experiment] = {}
        inversion_output[experiment]['stress_tensor'] =\
                np.zeros((3, 3), dtype=np.float32)
        inversion_output[experiment]['principal_stresses'] =\
                np.zeros(3, dtype=np.float32)
        inversion_output[experiment]['principal_directions'] =\
                np.zeros((3, 3), dtype=np.float32)
        inversion_output[experiment]['misfit'] = np.zeros(len(noise), dtype=np.float32)
        inversion_output[experiment]['boot_stress_tensor'] =\
                np.zeros((n_bootstraps, 3, 3), dtype=np.float32)
        inversion_output[experiment]['boot_principal_stresses'] =\
                np.zeros((n_bootstraps, 3), dtype=np.float32)
        inversion_output[experiment]['boot_principal_directions'] =\
                np.zeros((n_bootstraps, 3, 3), dtype=np.float32)
        inversion_output[experiment]['boot_misfit'] = np.zeros((n_bootstraps, len(noise)), dtype=np.float32)
    # -----------------------------------------
    # first, test the two inversion schemes on the true fault planes
    # ----------- whole data set
    inversion_output['true_fp_linear']['stress_tensor'],\
    inversion_output['true_fp_linear']['principal_stresses'],\
    inversion_output['true_fp_linear']['principal_directions'] =\
               utils_stress.Michael1984_inversion(
                       strikes, dips, rakes, return_eigen=True, return_stats=False,
                       Tarantola_kwargs=Tarantola_kwargs0)
    inversion_output['true_fp_linear']['misfit'] =\
            np.mean(utils_stress.mean_angular_residual(
                inversion_output['true_fp_linear']['stress_tensor'],
                strikes, dips, rakes))
    inversion_output['true_fp_iterative']['stress_tensor'], _,\
    inversion_output['true_fp_iterative']['principal_stresses'],\
    inversion_output['true_fp_iterative']['principal_directions'] =\
               utils_stress.Michael1984_iterative(
                       strikes, dips, rakes, return_eigen=True, return_stats=False,
                       Tarantola_kwargs=Tarantola_kwargs0, **Michael_kwargs)
    inversion_output['true_fp_iterative']['misfit'] =\
            np.mean(utils_stress.mean_angular_residual(
                inversion_output['true_fp_iterative']['stress_tensor'],
                strikes, dips, rakes))
    # ----------- bootstrapped data set
    for b in range(n_bootstraps):
        bootstrapped_set = np.random.choice(
                np.arange(n_earthquakes), replace=True, size=n_earthquakes)
        strikes_b, dips_b, rakes_b = strikes[bootstrapped_set], dips[bootstrapped_set], rakes[bootstrapped_set]
        inversion_output['true_fp_linear']['boot_stress_tensor'][b, ...],\
        inversion_output['true_fp_linear']['boot_principal_stresses'][b, ...],\
        inversion_output['true_fp_linear']['boot_principal_directions'][b, ...] =\
                   utils_stress.Michael1984_inversion(
                           strikes_b, dips_b, rakes_b, return_eigen=True, return_stats=False,
                           Tarantola_kwargs=Tarantola_kwargs0)
        inversion_output['true_fp_linear']['boot_misfit'][b, ...] =\
                np.mean(utils_stress.mean_angular_residual(
                    inversion_output['true_fp_linear']['boot_stress_tensor'][b, ...],
                    strikes_b, dips_b, rakes_b))
        inversion_output['true_fp_iterative']['boot_stress_tensor'][b, ...], _,\
        inversion_output['true_fp_iterative']['boot_principal_stresses'][b, ...],\
        inversion_output['true_fp_iterative']['boot_principal_directions'][b, ...] =\
                   utils_stress.Michael1984_iterative(
                           strikes_b, dips_b, rakes_b, return_eigen=True, return_stats=False,
                           Tarantola_kwargs=Tarantola_kwargs0, **Michael_kwargs)
        inversion_output['true_fp_iterative']['boot_misfit'][b, ...] =\
                np.mean(utils_stress.mean_angular_residual(
                    inversion_output['true_fp_iterative']['boot_stress_tensor'][b, ...],
                    strikes_b, dips_b, rakes_b))
    Tarantola_kwargs = {}
    for n in range(len(noise)):
        Tarantola_kwargs['noise{:d}'.format(n)] = Tarantola_kwargs0.copy()
        print(f'Noise level {n}, linear inversion...')
        # simple, linear inversion
        inversion_output['linear']['stress_tensor'][n, ...],\
        inversion_output['linear']['principal_stresses'][n, ...],\
        inversion_output['linear']['principal_directions'][n, ...] =\
                utils_stress.inversion_one_set_nodal_planes(
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        n_random_selections=n_random_selections,
                        **Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        iterative_method=False)
        print(f'Noise level {n}, non-linear iterative inversion...')
        # non-linear inversion for both the stress tensor and shear stresses
        inversion_output['iterative']['stress_tensor'][n, ...],\
        inversion_output['iterative']['principal_stresses'][n, ...],\
        inversion_output['iterative']['principal_directions'][n, ...] =\
                utils_stress.inversion_one_set_nodal_planes(
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        n_random_selections=n_random_selections,
                        **Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        iterative_method=True)
        print(f'Noise level {n}, failure criterion inversion...')
        inversion_output['failure_criterion']['stress_tensor'][n, ...],\
        inversion_output['failure_criterion']['friction'][n],\
        inversion_output['failure_criterion']['principal_stresses'][n, ...],\
        inversion_output['failure_criterion']['principal_directions'][n, ...] =\
                utils_stress.inversion_one_set_instability(
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        n_random_selections=n_random_selections,
                        Michael_kwargs=Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        friction_min=friction_min,
                        friction_max=friction_max,
                        friction_step=friction_step,
                        n_stress_iter=n_stress_iter,
                        n_averaging=n_averaging,
                        iterative_method=False)
        print(f'Noise level {n}, non-linear iterative and failure '
              'criterion inversion...')
        inversion_output['iterative_failure_criterion']['stress_tensor'][n, ...],\
        inversion_output['iterative_failure_criterion']['friction'][n],\
        inversion_output['iterative_failure_criterion']['principal_stresses'][n, ...],\
        inversion_output['iterative_failure_criterion']['principal_directions'][n, ...] =\
                utils_stress.inversion_one_set_instability(
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        n_random_selections=n_random_selections,
                        Michael_kwargs=Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        friction_min=friction_min,
                        friction_max=friction_max,
                        friction_step=friction_step,
                        n_stress_iter=n_stress_iter,
                        n_averaging=n_averaging,
                        plot=False)
        for method in methods:
            inversion_output[method]['misfit'][n] =\
                    np.mean(utils_stress.mean_angular_residual(
                        inversion_output[method]['stress_tensor'][n, ...],
                        strikes, dips, rakes))
        print(f'Noise level {n}, linear inversion (bootstrapping)...')
        # simple, linear inversion
        inversion_output['linear']['boot_stress_tensor'][n, ...],\
        inversion_output['linear']['boot_principal_stresses'][n, ...],\
        inversion_output['linear']['boot_principal_directions'][n, ...] =\
                utils_stress.inversion_bootstrap_nodal_planes(
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        n_bootstraps=n_bootstraps,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        iterative_method=False)
        print(f'Noise level {n}, non-linear iterative inversion (bootstrapping)...')
        # non-linear inversion for both the stress tensor and shear stresses
        inversion_output['iterative']['boot_stress_tensor'][n, ...],\
        inversion_output['iterative']['boot_principal_stresses'][n, ...],\
        inversion_output['iterative']['boot_principal_directions'][n, ...] =\
                utils_stress.inversion_bootstrap_nodal_planes(
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        n_bootstraps=n_bootstraps,
                        Michael_kwargs=Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        iterative_method=True)
        print(f'Noise level {n}, failure criterion inversion (bootstrapping)...')
        inversion_output['failure_criterion']['boot_stress_tensor'][n, ...],\
        inversion_output['failure_criterion']['boot_principal_stresses'][n, ...],\
        inversion_output['failure_criterion']['boot_principal_directions'][n, ...] =\
                utils_stress.inversion_bootstrap_instability(
                        inversion_output['failure_criterion']['principal_directions'][n, ...],
                        utils_stress.R_(inversion_output['failure_criterion']['principal_stresses'][n, ...]),
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        inversion_output['failure_criterion']['friction'][n],
                        Michael_kwargs=Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        n_stress_iter=n_stress_iter,
                        n_bootstraps=n_bootstraps,
                        iterative_method=False, verbose=0)
        print(f'Noise level {n}, non-linear iterative and failure '
              'criterion inversion (bootstrapping)...')
        inversion_output['iterative_failure_criterion']['boot_stress_tensor'][n, ...],\
        inversion_output['iterative_failure_criterion']['boot_principal_stresses'][n, ...],\
        inversion_output['iterative_failure_criterion']['boot_principal_directions'][n, ...] =\
                utils_stress.inversion_bootstrap_instability(
                        inversion_output['iterative_failure_criterion']['principal_directions'][n, ...],
                        utils_stress.R_(inversion_output['iterative_failure_criterion']['principal_stresses'][n, ...]),
                        strikes_1[:, n], dips_1[:, n], rakes_1[:, n],
                        strikes_2[:, n], dips_2[:, n], rakes_2[:, n],
                        inversion_output['iterative_failure_criterion']['friction'][n],
                        Michael_kwargs=Michael_kwargs,
                        Tarantola_kwargs=Tarantola_kwargs['noise{:d}'.format(n)],
                        n_stress_iter=n_stress_iter,
                        n_bootstraps=n_bootstraps,
                        iterative_method=True, verbose=0)
        for method in methods:
            for b in range(n_bootstraps):
                inversion_output[method]['boot_misfit'][n, b] =\
                        np.mean(utils_stress.mean_angular_residual(
                            inversion_output[method]['boot_stress_tensor'][n, b, ...],
                            strikes, dips, rakes))
    inversion_output['strikes'] = np.stack((strikes_1, strikes_2), axis=2)
    inversion_output['dips'] = np.stack((dips_1, dips_2), axis=2)
    inversion_output['rakes'] = np.stack((rakes_1, rakes_2), axis=2)
    return true_stress_tensor, inversion_output

def rotation_x1(theta):
    theta = theta*np.pi/180.
    R = np.array([[1., 0., 0.],
                  [0., np.cos(theta), -np.sin(theta)],
                  [0., np.sin(theta), np.cos(theta)]])
    return R

def rotation_x3(theta):
    theta = theta*np.pi/180.
    R = np.array([[np.cos(theta), -np.sin(theta), 0.],
                  [np.sin(theta), np.cos(theta), 0.],
                  [0., 0., 1.]])
    return R

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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    from matplotlib.colors import LinearSegmentedColormap
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_inverted_stress_tensors(true_stress_tensor, inversion_output, axes=None, **kwargs):
    hist_kwargs = {}
    hist_kwargs['smoothing_sig'] = kwargs.get('smoothing_sig', 1)
    hist_kwargs['nbins'] = kwargs.get('nbins', 200)
    hist_kwargs['return_count'] = kwargs.get('return_count', True)
    hist_kwargs['confidence_intervals'] = kwargs.get('confidence_intervals', [95.])
    cmaps = ['Blues', 'Oranges', 'Greens', 'Reds']
    cmaps = [truncate_colormap(plt.get_cmap(cmap), minval=0., maxval=0.75) for cmap in cmaps]
    n_bootstraps = inversion_output['linear']['boot_principal_directions'][0, ...].shape[0]
    true_ps, true_pd = utils_stress.stress_tensor_eigendecomposition(true_stress_tensor)
    true_R = utils_stress.R_(true_ps)
    markers = ['o', 's', 'v']
    methods = ['linear', 'failure_criterion', 'iterative', 'iterative_failure_criterion']
    fig = plt.figure('inverted_stress_tensors', figsize=(18, 9))
    ax = fig.add_subplot(3, 4, 1, projection='stereonet')
    ax.set_title('True fault planes', pad=30)
    ax.plane(inversion_output['strikes'][:, 0, 0],
             inversion_output['dips'][:, 0, 0], color='k',
             lw=1.0)
    ax1 = fig.add_subplot(3, 4, 5, projection='stereonet')
    for i in range(3):
        az, pl = utils_focmech.get_bearing_plunge(true_pd[:, i])
        ax1.line(pl, az, marker=markers[i], markeredgecolor='k',
                 markeredgewidth=2, markerfacecolor='none',
                 markersize=20, label=r'True $\sigma_{{{:d}}}$: {:.1f}'u'\u00b0''|{:.1f}'u'\u00b0'.
                 format(i+1, az%360., pl))
    for method, cl, cmap in zip(['linear', 'iterative'], [_colors_[0], _colors_[2]], [cmaps[0], cmaps[2]]):
        exp = f'true_fp_{method}'
        R = utils_stress.R_(inversion_output[exp]['principal_stresses'])
        for k in range(3):
            if k == 0:
                label = '{}, R={:.2f}, $\\vert{{\\Delta \\theta}}\\vert$={:.1f}'u'\u00b0'.\
                        format(method.capitalize(), R, inversion_output[exp]['misfit'])
            else:
                label = ''
            az, pl = utils_focmech.get_bearing_plunge(
                    inversion_output[exp]['principal_directions'][:, k])
            ax1.line(pl, az, marker=markers[k], markeredgecolor=cl, markerfacecolor='none',
                     markeredgewidth=2, markersize=15, label=label)
            boot_pd_stereo = np.zeros((n_bootstraps, 2), dtype=np.float32)
            for b in range(n_bootstraps):
                boot_pd_stereo[b, :] = utils_focmech.get_bearing_plunge(
                        inversion_output[exp]['boot_principal_directions'][b, :, k])
            #ax1.line(boot_pd_stereo[:, 1], boot_pd_stereo[:, 0], marker=markers[k],
            #         markersize=5, color=cl, alpha=0.30, zorder=0.99)
            count, lons_g, lats_g, levels = get_CI_levels(
                    boot_pd_stereo[:, 0], boot_pd_stereo[:, 1], **hist_kwargs)
            ax1.contour(lons_g, lats_g, count, levels=levels, vmin=0., colors=cl)
    fake_handle = [ax1.plot([], [], marker='', ls='')[0]]
    title_label = ['True stress tensor (R={:.2f}):'.format(true_R)]
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend(fake_handle+handles, title_label+labels, loc='upper left', bbox_to_anchor=(-0.1, -0.15))
    axes = [ax1]
    titles = ['Noise-free', 'Low Noise', 'High Noise']
    for i, title in enumerate(titles):
        ax1 = fig.add_subplot(3, 4, 2+i,projection='stereonet')
        ax1.set_title(title, pad=30)
        ax1.plane(inversion_output['strikes'][:, i, 0],
                  inversion_output['dips'][:, i, 0], color='k',
                  lw=1.0)
        ax1.plane(inversion_output['strikes'][:, i, 1],
                  inversion_output['dips'][:, i, 1], color='dimgray',
                  lw=0.65)
        ax = fig.add_subplot(3, 4, 6+i, projection='stereonet')
        for j, method in enumerate(methods):
            R = utils_stress.R_(inversion_output[method]['principal_stresses'][i, ...])
            for k in range(3):
                az, pl = utils_focmech.get_bearing_plunge(true_pd[:, k])
                ax.line(pl, az, marker=markers[k], markeredgecolor='k', markerfacecolor='none',
                        markeredgewidth=2, zorder=1, markersize=20)
                if k == 0:
                    label = '{}:\nR={:.2f}, $\\vert{{\\Delta \\theta}}\\vert$={:.1f}'u'\u00b0'.\
                            format(method.replace('_', ' ').capitalize(), R, inversion_output[method]['misfit'][i])
                else:
                    label = ''
                az, pl = utils_focmech.get_bearing_plunge(
                        inversion_output[method]['principal_directions'][i, :, k])
                ax.line(pl, az, marker=markers[k], markeredgecolor=_colors_[j], markerfacecolor='none',
                        markeredgewidth=2, markersize=15, label=label, zorder=2)
                boot_pd_stereo = np.zeros((n_bootstraps, 2), dtype=np.float32)
                for b in range(n_bootstraps):
                    boot_pd_stereo[b, :] = utils_focmech.get_bearing_plunge(
                            inversion_output[method]['boot_principal_directions'][i, b, :, k])
                #ax.line(boot_pd_stereo[:, 1], boot_pd_stereo[:, 0], marker=markers[k],
                #        markersize=5, color=_colors_[j], alpha=0.30, zorder=0.99)
                count, lons_g, lats_g, levels = get_CI_levels(
                        boot_pd_stereo[:, 0], boot_pd_stereo[:, 1], **hist_kwargs)
                ax.contour(lons_g, lats_g, count, levels=levels, vmin=0., colors=_colors_[j])
                #ax.contour(lons_g, lats_g, count, levels=levels, vmin=0., cmap=cmaps[j])
        ax.legend(loc='upper left', bbox_to_anchor=(-0.1, -0.15))
        axes.append(ax)
        #ax3 = fig.add_subplot(4, 4, 10+i)
        #Rs = np.zeros(n_bootstraps, dtype=np.float32)
        #for b in range(n_bootstraps):
        #    Rs[b] = utils_stress.R_(inversion_output[method]['boot_principal_stresses'][i, b, :])
        #ax3.hist(Rs, range=(0., 1.), bins=20, color=_colors_[j], histtype='step')
        #ax3.set_xlabel('Shape Ratio')
        #ax3.set_ylabel('Count')
    #for ax in axes:
    #    #ax.grid(True)
    plt.subplots_adjust(top=0.93, bottom=0.11,
                        left=0.05, right=0.95,
                        hspace=0.25, wspace=0.4)
    return fig

def plot_shape_ratios(true_stress_tensor, inversion_output):
    true_ps, true_pd = utils_stress.stress_tensor_eigendecomposition(true_stress_tensor)
    true_R = utils_stress.R_(true_ps)
    fig = plt.figure('shape_ratios', figsize=(18, 9))
    titles = ['Noise-free', 'Low Noise', 'High Noise']
    methods = ['linear', 'failure_criterion', 'iterative', 'iterative_failure_criterion']
    n_bootstraps = inversion_output['linear']['boot_stress_tensor'].shape[1]
    ax = fig.add_subplot(3, 4, 1)
    for exp, cl in zip(['true_fp_linear', 'true_fp_iterative'], [_colors_[0], _colors_[2]]):
        Rs = np.zeros(n_bootstraps, dtype=np.float32)
        for b in range(n_bootstraps):
            Rs[b] = utils_stress.R_(inversion_output[exp]['boot_principal_stresses'][b, :])
        ax.hist(Rs, range=(0., 1.), bins=20, lw=2.5, color=cl, histtype='step')
        ax.axvline(true_R, color='k')
        ax.set_xlabel('Shape Ratio')
        ax.set_ylabel('Count')
    for i, title in enumerate(titles):
        ax = fig.add_subplot(3, 4, 2+i)
        for j, method in enumerate(methods):
            R = utils_stress.R_(inversion_output[method]['principal_stresses'][i, ...])
            Rs = np.zeros(n_bootstraps, dtype=np.float32)
            for b in range(n_bootstraps):
                Rs[b] = utils_stress.R_(inversion_output[method]['boot_principal_stresses'][i, b, :])
            ax.hist(Rs, range=(0., 1.), bins=20, lw=2.5, color=_colors_[j], histtype='step')
            ax.axvline(true_R, color='k')
            ax.set_xlabel('Shape Ratio')
            ax.set_ylabel('Count')
    plt.subplots_adjust(top=0.93, bottom=0.11,
                        left=0.06, right=0.95,
                        hspace=0.25, wspace=0.4)
    return fig


def plot_shear_magnitudes(true_stress_tensor, inversion_output, axes=None):
    normals, slips = utils_stress.normal_slip_vectors(
            inversion_output['strikes'][..., 0],
            inversion_output['dips'][..., 0],
            inversion_output['rakes'][..., 0])
    _, _, true_shear = utils_stress.compute_traction(
            true_stress_tensor, normals[:, :, 0].T)
    _, _, true_fp_linear_shear = utils_stress.compute_traction(
            inversion_output['true_fp_linear']['stress_tensor'], normals[:, :, 0].T)
    _, _, true_fp_iterative_shear = utils_stress.compute_traction(
            inversion_output['true_fp_iterative']['stress_tensor'], normals[:, :, 0].T)
    fig = plt.figure('shear', figsize=(18, 9))
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.scatter(np.sqrt(np.sum(true_shear**2, axis=-1)),
                np.sqrt(np.sum(true_fp_linear_shear**2, axis=-1)),
                color=_colors_[0], label='Naive')
    ax1.scatter(np.sqrt(np.sum(true_shear**2, axis=-1)),
                np.sqrt(np.sum(true_fp_iterative_shear**2, axis=-1)),
                color=_colors_[2], label='Iterative')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('True shear stress $\\vert \\tau \\vert$')
    ax1.set_ylabel('Predicted shear stress $\\vert \\tilde{\\tau} \\vert$')
    # -------------------
    ax2 = fig.add_subplot(2, 4, 5)
    ax2.hist(np.sqrt(np.sum(true_shear**2, axis=-1)), range=(0., 1.0), bins=20, color='k', histtype='step', lw=2.5, label='True')
    ax2.hist(np.sqrt(np.sum(true_fp_linear_shear**2, axis=-1)), range=(0., 1.0), bins=20, color=_colors_[0], alpha=0.5, label='Naive')
    ax2.hist(np.sqrt(np.sum(true_fp_iterative_shear**2, axis=-1)), range=(0., 1.0), bins=20, color=_colors_[2], alpha=0.5, label='Iterative')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Shear stress $\\vert \\tau \\vert$')
    ax2.legend(loc='best')
    # -------------------
    plt.subplots_adjust(top=0.98, bottom=0.15, left=0.10, right=0.98)
    return fig

def plot_fault_planes(inversion_output, axes=None):
    fig = plt.figure('fault_planes', figsize=(18, 9))
    # plot true fault planes
    ax = fig.add_subplot(3, 4, 1, projection='stereonet')
    ax.set_title('True fault planes', pad=30)
    ax.plane(inversion_output['strikes'][:, 0, 0],
             inversion_output['dips'][:, 0, 0], color='k', lw=1.0)
    titles = ['Noise-free', 'Low Noise', 'High Noise']
    for i, title in enumerate(titles):
        ax1 = fig.add_subplot(3, 4, 2+i,projection='stereonet')
        ax1.set_title(title, pad=30)
        ax1.plane(inversion_output['strikes'][:, i, 0],
                  inversion_output['dips'][:, i, 0], color='k', lw=1.0)
        ax1.plane(inversion_output['strikes'][:, i, 1],
                  inversion_output['dips'][:, i, 1], color='dimgray', lw=0.65)
        # get predicted fault planes for failure criterion method
        for j, (method, cl) in enumerate(zip(['failure_criterion', 'iterative_failure_criterion'], [_colors_[1], _colors_[3]])):
            R = utils_stress.R_(inversion_output[method]['principal_stresses'][i, ...])
            I, fp_strikes, fp_dips, fp_rakes = utils_stress.compute_instability_parameter(
                    inversion_output[method]['principal_directions'][i, ...], R, inversion_output[method]['friction'][i],
                    inversion_output['strikes'][:, i, 0], inversion_output['dips'][:, i, 0], inversion_output['rakes'][:, i, 0],
                    inversion_output['strikes'][:, i, 1], inversion_output['dips'][:, i, 1], inversion_output['rakes'][:, i, 1],
                    return_fault_planes=True)
            ax2 = fig.add_subplot(3, 4, 2+4*(j+1)+i, projection='stereonet')
            ax2.plane(fp_strikes, fp_dips, color=cl, lw=1.0)
            ax2.plane(fp_strikes[0], fp_dips[0], color=cl, lw=1.0,
                      label=method.replace('_', ' ').capitalize())
            if i == 0:
                ax2.legend(loc='upper right', bbox_to_anchor=(-0.25, 0.65))
    plt.subplots_adjust(top=0.88, bottom=0.11,
                        left=0.05, right=0.85,
                        hspace=0.35, wspace=0.2)
    return fig

def plot_summary(true_stress_tensor, inversion_output, figname='',
                 ncols=4, nrows=5, figsize=(18, 9)):
    fig = plt.figure(figname, figsize=figsize)
    gs = fig.add_gridspec(ncols=ncols, nrows=nrows)
    stereo_axes = [[0,0], [0,1], [0,2], [0,3],
                   [1,0], [1,1], [1,2], [1,3],
                   [3,1], [3,2], [3,3],
                   [4,1], [4,2], [4,3]]
    regular_axes = [[3,0], [4,0]]
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            if [i,j] in stereo_axes:
                axes[i,j] = fig.add_subplot(gs[i, j], projection='stereonet')
            elif [i,j] in regular_axes:
                axes[i,j] = fig.add_subplot(gs[i, j])
    fig = plot_inverted_stress_tensors(
            true_stress_tensor, inversion_output, axes=axes[1:2, :])
    fig = plot_shear_magnitudes(
            true_stress_tensor, inversion_output, axes=axes[3:, 0:1])
    fig = plot_fault_planes(
            inversion_output, axes=axes[np.array([0,3,4]), :])
    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.05, right=0.95)
    return fig

def plot_focal_mechanisms(inversion_output):
    from obspy.imaging.beachball import beach
    n_earthquakes = inversion_output['strikes'].shape[0]
    n = int(np.sqrt(n_earthquakes)+1)
    fig = plt.figure('focal_mechanisms', figsize=(18, 9))
    ax = fig.add_subplot(111)
    W = 10
    for i in range(n_earthquakes):
        fm = [inversion_output['strikes'][i, 0, 0],
              inversion_output['dips'][i, 0, 0],
              inversion_output['rakes'][i, 0, 0]]
        bb = beach(fm, xy=(1.5*W*(i%n), 1.5*W*(i//n)), facecolor='k', width=W)
        ax.add_collection(bb)
    ax.set_xlim(-W, 1.5*n*W)
    ax.set_ylim(-W, 1.5*n*W)
    ax.set_aspect('equal')
    return fig

def plot_PT_axes(true_stress_tensor, inversion_output):
    # get the optimally oriented fault plane?
    mt = pmt.MomentTensor(mnn=true_stress_tensor[0,0], mee=true_stress_tensor[1,1],
                          mdd=true_stress_tensor[2,2], mne=-true_stress_tensor[0,1],
                          mnd=-true_stress_tensor[0,2], med=true_stress_tensor[1,2])
    [s1, d1, r1], [s2, d2, r2] = mt.both_strike_dip_rake()
    # get the P and T axes in the (north, east, down) coordinate system
    P_axis0, T_axis0 = mt.p_axis().A[0], mt.t_axis().A[0]
    # convert to my coordinate system: (north, west, up)
    P_axis0 *= np.array([1., -1., -1.])
    T_axis0 *= np.array([1., -1., -1.])
    # compute all P/T axes
    n_earthquakes = inversion_output['strikes'].shape[0]
    P_axis = np.zeros((n_earthquakes, 2), dtype=np.float32)
    T_axis = np.zeros((n_earthquakes, 2), dtype=np.float32)
    N_axis = np.zeros((n_earthquakes, 2), dtype=np.float32)
    r2d = 180./np.pi
    for t in range(n_earthquakes):
        # first, get normal and slip vectors from
        # strike, dip, rake
        normal, slip = utils_focmech.normal_slip_vectors(
                inversion_output['strikes'][t, 0, 0],
                inversion_output['dips'][t, 0, 0],
                inversion_output['rakes'][t, 0, 0])
        # second, get the t and p vectors
        p_axis, t_axis, n_axis = utils_focmech.p_t_n_axes(normal, slip)
        p_bearing, p_plunge = utils_focmech.get_bearing_plunge(p_axis)
        t_bearing, t_plunge = utils_focmech.get_bearing_plunge(t_axis)
        n_bearing, n_plunge = utils_focmech.get_bearing_plunge(n_axis)
        P_axis[t, :] = p_bearing, p_plunge
        T_axis[t, :] = t_bearing, t_plunge
        N_axis[t, :] = n_bearing, n_plunge
    fig = plt.figure('PT_axes', figsize=(18, 9))
    ax = fig.add_subplot(2, 2, 1, projection='stereonet')
    ax.line(T_axis[:, 1], T_axis[:, 0], marker='o', color='C0')
    ax.line(P_axis[:, 1], P_axis[:, 0], marker='v', color='C3')
    ax.line(T_axis[0, 1], T_axis[0, 0], marker='o', color='C0', label='T axis')
    ax.line(P_axis[0, 1], P_axis[0, 0], marker='v', color='C3', label='P axis')
    t_bearing, t_plunge = utils_focmech.get_bearing_plunge(T_axis0)
    ax.line(t_plunge, t_bearing, marker='o', markersize=10, color='k')
    p_bearing, p_plunge = utils_focmech.get_bearing_plunge(P_axis0)
    ax.line(p_plunge, p_bearing, marker='v', markersize=10, color='k')
    ax.plane([s1, s2], [d1, d2], color='k')
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(1.0, 1.0))
    return fig

def save_results(true_stress_tensor, inversion_output, filename):
    import h5py as h5
    with h5.File(filename, mode='w') as f:
        for key1 in inversion_output.keys():
            if isinstance(inversion_output[key1], dict):
                f.create_group(key1)
                for key2 in inversion_output[key1].keys():
                    f[key1].create_dataset(key2, data=inversion_output[key1][key2])
            else:
                f.create_dataset(key1, data=inversion_output[key1])
        f.create_dataset('true_stress_tensor', data=true_stress_tensor)

def read_results(filename):
    import h5py as h5
    inversion_output = {}
    with h5.File(filename, mode='r') as f:
        for key1 in f.keys():
            if isinstance(f[key1], h5.Dataset):
                inversion_output[key1] = f[key1][()]
            else:
                inversion_output[key1] = {}
                for key2 in f[key1].keys():
                    inversion_output[key1][key2] = f[key1][key2][()]
    return inversion_output['true_stress_tensor'], inversion_output

if __name__ == "__main__":
    experiment = 2
    #true_stress_tensor, strikes, dips, rakes, G, shear_forward, shear_direct, n_ = \
    #        test_forward_model()
    #ps, pd = utils_stress.stress_tensor_eigendecomposition(true_stress_tensor)
    #planes2 = [tuple() + utils_focmech.aux_plane(strikes[i], dips[i], rakes[i])
    #           for i in range(len(strikes))]
    #s2 = np.array([planes2[i][0] for i in range(len(strikes))])
    #d2 = np.array([planes2[i][0] for i in range(len(dips))])
    #r2 = np.array([planes2[i][0] for i in range(len(rakes))])
    #I, fs, fd, fr = utils_stress.compute_instability_parameter(
    #        pd, utils_stress.R_(ps), 0.70, strikes, dips, rakes, s2, d2, r2, return_fault_planes=True)
    #true_stress_tensor, inversion_output = main(experiment=experiment)
    true_stress_tensor, inversion_output = read_results(f'./experiment{experiment}.h5')
    fig1 = plot_inverted_stress_tensors(true_stress_tensor, inversion_output, smoothing_sig=2)
    fig2 = plot_fault_planes(inversion_output)
    fig3 = plot_shear_magnitudes(true_stress_tensor, inversion_output)
    #fig4 = plot_summary(true_stress_tensor, inversion_output)
    #fig5 = plot_focal_mechanisms(inversion_output)
    #fig6 = plot_PT_axes(true_stress_tensor, inversion_output)
    fig7 = plot_shape_ratios(true_stress_tensor, inversion_output)
    #save_results(true_stress_tensor, inversion_output, f'./experiment{experiment}.h5')
