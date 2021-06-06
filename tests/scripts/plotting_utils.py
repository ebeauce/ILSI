import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.colorbar as clb
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

import mplstereonet

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

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

