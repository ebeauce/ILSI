import sys

import numpy as np
from numpy.linalg import LinAlgError

def hist2d(azimuths, plunges,
           nbins=200, smoothing_sig=0,
           plot=False):
    """
    Computes the 2d histogram in the stereographic space
    of a collection lines described by their azimuth and plunge.

    Parameters
    -----------
    azimuths: (n_lines) list or array, float
        Azimuths of the lines.
    plunges: (n_lines) list or array, float
        Plunges (angle from horizontal) of the lines.
    nbins: integer, default to 200
        Number of bins, in both axes, used to discretized
        the 2d space.
    smoothing_sih: float, default to 0
        If greater than 0, smooth the 2d distribution
        with a gaussian kernel. This is useful to derive
        smooth confidence intervals.
    plot: boolean, default to False
        If True, plot the 2d histogram.
    """
    import mplstereonet
    from scipy.ndimage.filters import gaussian_filter
    # convert azimuths and plunges to longitudes and latitudes
    # on a stereographic plot
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
    """
    Computes the 2d histogram in the stereographic space
    of a collection lines described by their azimuth and plunge.

    Parameters
    -----------
    azimuths: (n_lines) list or array, float
        Azimuths of the lines.
    plunges: (n_lines) list or array, float
        Plunges (angle from horizontal) of the lines.
    nbins: integer, default to 200
        Number of bins, in both axes, used to discretized
        the 2d space.
    smoothing_sih: float, default to 1
        If greater than 0, smooth the 2d distribution
        with a gaussian kernel. This is useful to derive
        smooth confidence intervals.
    plot: boolean, default to False
        If True, plot the 2d histogram.
    """
    from scipy.interpolate import interp1d
    # get histogram on a 2d grid
    count, lons_g, lats_g = hist2d(
            azimuths, plunges, nbins=nbins, smoothing_sig=smoothing_sig)
    # flatten the count array and sort it from largest to smallest
    count_vector = np.sort(count.copy().flatten())[::-1]
    # compute the "CDF" of the counts
    count_CDF = np.hstack(
            ([0.], np.cumsum(count_vector)/count_vector.sum(), [1.]))
    count_vector = np.hstack(([count_vector.max()+1.], count_vector, [0.]))
    # build an interpolator that gives the count number 
    # for a given % of the total mass
    mass_dist = interp1d(100.*count_CDF, count_vector)
    mass_dist_ = lambda x: mass_dist(x).item()
    confidence_intervals = list(
            map(mass_dist_, [k for k in confidence_intervals]))
    if plot:
        fig = plt.figure('2d_histogram_stereo', figsize=(18, 9))
        ax = fig.add_subplot(111, projection='stereonet')
        pcl = ax.pcolormesh(lons_g, lats_g, count)
        ax.contour(
                lons_g, lats_g, count,
                levels=confidence_intervals, zorder=2, cmap='jet')
        plt.colorbar(mappable=pcl)
    if return_count:
        return count, lons_g, lats_g, confidence_intervals
    else:
        return confidence_intervals


def angular_residual(stress_tensor, strikes, dips, rakes):
    angles = np.zeros(len(strikes), dtype=np.float32)
    for i in range(len(strikes)):
        angles[i] = shear_slip_angle_difference(
                stress_tensor, strikes[i], dips[i], rakes[i])
    return angles

def aux_plane(s1, d1, r1):
    """
    Get Strike and dip of second plane.
    
    Adapted from MATLAB script
    `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
    written by Andy Michael, Chen Ji and Oliver Boyd.

    TAKEN FROM OBSPY's SOURCE CODE
    """
    r2d = 180 / np.pi

    def _strike_dip(n, e, u):
        """
        Finds strike and dip of plane given normal vector having components n, e,
        and u.
        
        Adapted from MATLAB script
        `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
        written by Andy Michael, Chen Ji and Oliver Boyd.
        """
        r2d = 180 / np.pi
        if u < 0:
            n = -n
            e = -e
            u = -u

        strike = np.arctan2(e, n) * r2d
        strike = strike - 90
        while strike >= 360:
            strike = strike - 360
        while strike < 0:
            strike = strike + 360
        x = np.sqrt(np.power(n, 2) + np.power(e, 2))
        dip = np.arctan2(x, u) * r2d
        return (strike, dip)

    # modified by me:
    if r1 > 180.:
        # convert rake between 0 and 360
        # to rake between -180 and +180
        r1 = r1-360.
    
    z = (s1 + 90) / r2d
    z2 = d1 / r2d
    z3 = r1 / r2d
    # slick vector in plane 1
    sl1 = -np.cos(z3) * np.cos(z) - np.sin(z3) * np.sin(z) * np.cos(z2)
    sl2 = np.cos(z3) * np.sin(z) - np.sin(z3) * np.cos(z) * np.cos(z2)
    sl3 = np.sin(z3) * np.sin(z2)
    (strike, dip) = _strike_dip(sl2, sl1, sl3)
    
    n1 = np.sin(z) * np.sin(z2)  # normal vector to plane 1
    n2 = np.cos(z) * np.sin(z2)
    h1 = -sl2  # strike vector of plane 2
    h2 = sl1
    # note h3=0 always so we leave it out
    # n3 = np.cos(z2)
    
    z = h1 * n1 + h2 * n2
    z = z / np.sqrt(h1 * h1 + h2 * h2)
    # we might get above 1.0 only due to floating point
    # precision. Clip for those cases.
    float64epsilon = 2.2204460492503131e-16
    if 1.0 < abs(z) < 1.0 + 100 * float64epsilon:
        z = np.copysign(1.0, z)
    z = np.arccos(z)
    rake = 0
    if sl3 > 0:
        rake = z * r2d
    if sl3 <= 0:
        rake = -z * r2d
    return strike%360., dip, rake%360.

def compute_traction(stress_tensor, normal):
    """
    Parameters
    -----------
    stress_tensor: (3, 3) numpy array
        Cauchy stress tensor.
    normal: (n_earthquakes, 3) numpy array
        Matrix of n_earthquakes row vectors of fault normals.

    Returns
    --------
    traction: (n_earthquakes, 3) numpy array
        Tractions on the surfaces defined by normal.
    normal_traction: (n_earthquakes, 3) numpy array
        Normal component of the tractions.
    shear_traction: (n_earthquakes, 3) numpy array
        Tangential component of the tractions.
    """
    traction = np.dot(stress_tensor, normal.T).T
    normal_traction = np.sum(traction*normal, axis=-1)[:, np.newaxis]\
                      *normal
    shear_traction = traction - normal_traction
    return traction, normal_traction, shear_traction

def errors_in_data(strike, dip, rake,
                   jack_strikes_1, jack_dips_1, jack_rakes_1,
                   jack_strikes_2, jack_dips_2, jack_rakes_2):
    """
    Use the multiple solutions obtained during the jackknife resampling
    of the focal mechanism inversion to compute the deviation of these
    multiple solutions from the best solution. We are interested in computing
    the deviation of the three components (north, west, vertical) of the slip
    vectors from the best slip vector since the slip vectors are used in
    the stress inversion. Because there are two possible slip vectors
    for each focal mechanism solution, we systematically look among the
    jackknife solutions 1 and 2 for the closest slip vector to the target
    vector, defined by (strike, dip, rake).
    We recommend to run this function for (strike, dip, rake)_1 and
    (strike, dip, rake)_2 of the best focal mechanism solution, and
    average the outputs.
    """
    r2d = 180./np.pi
    n_jackknife = len(jack_strikes_1)
    # slip vector from the best nodal plane
    _, slip_vector_best = normal_slip_vectors(strike, dip, rake)
    # slip vectors from the jackknife nodal planes
    slip_vectors = np.zeros((n_jackknife, 2, 3), dtype=np.float32)
    # angles between the jackknife slip vectors and the best slip vector
    # since it is ambiguous which of the planes are the fault planes,
    # we simply systematically compute the angle between both planes
    # and keep the lowest angle.
    slip_angles = np.zeros(n_jackknife, dtype=np.float32)
    for i in range(n_jackknife):
        _, slip_vectors[i, 0, :] = normal_slip_vectors(
                jack_strikes_1[i], jack_dips_1[i], jack_rakes_1[i])
        _, slip_vectors[i, 1, :] = normal_slip_vectors(
                jack_strikes_2[i], jack_dips_2[i], jack_rakes_2[i])
    # a little bit of clipping is necessary in case of numerical errors
    # putting the scalar products sligthly above or below +1/-1.
    scalar_prod1 = np.clip(np.sum(slip_vectors[:, 0, :]*slip_vector_best, axis=-1), -1., +1.)
    scalar_prod2 = np.clip(np.sum(slip_vectors[:, 1, :]*slip_vector_best, axis=-1), -1., +1.)
    angles_1 = np.arccos(scalar_prod1)
    angles_2 = np.arccos(scalar_prod2)
    abs_angles_1 = np.abs(np.arccos(scalar_prod1))
    abs_angles_2 = np.abs(np.arccos(scalar_prod2))
    slip_angles = np.minimum(abs_angles_1, abs_angles_2)*r2d
    mask1 = abs_angles_1 < abs_angles_2
    slip_angles[mask1] *= np.sign(angles_1[mask1])
    mask2 = abs_angles_2 <= abs_angles_1
    slip_angles[mask2] *= np.sign(angles_2[mask2])
    slip_vectors_ = np.zeros((n_jackknife, 3), dtype=np.float32)
    # get the closest slip vectors
    slip_vectors_[mask1, :] = slip_vectors[mask1, 0, :]
    slip_vectors_[mask2, :] = slip_vectors[mask2, 1, :]
    # we can now use the standard deviations on each of the
    # 3 components to estimate errors in the data and use
    # Tarantola and Valette formula
    dev_n = 1.42*np.median(np.abs(slip_vectors_[:, 0] - slip_vector_best[0]))
    dev_w = 1.42*np.median(np.abs(slip_vectors_[:, 1] - slip_vector_best[1]))
    dev_z = 1.42*np.median(np.abs(slip_vectors_[:, 2] - slip_vector_best[2]))
    #for i in range(3):
    #    plt.hist(slip_vectors_[:, i], bins=20)
    #    plt.axvline(slip_vector_best[i], lw=2, color='C{:d}'.format(i))
    #plt.show()
    return dev_n, dev_w, dev_z

def get_bearing_plunge(u, degrees=True, hemisphere='lower'):
    """
    The vectors are in the coordinate system (x1, x2, x3):
    x1: north
    x2: west
    x3: upward
    """
    r2d = 180./np.pi
    if hemisphere == 'lower' and u[2] > 0.:
        # we need to consider the end of the line
        # that plunges downward and crosses the
        # lower hemisphere
        u = -1.*u
    elif hemisphere == 'upper' and u[2] < 0.:
        u = -1.*u
    # the trigonometric sense is the opposite of the azimuthal sense,
    # therefore we need a -1 multiplicative factor
    bearing = -1.*np.arctan2(u[1], u[0])
    # the plunge is measured downward from the end of the
    # line specified by the bearing
    # this formula is valid for p_axis[2] < 0
    plunge = (np.arccos(u[2]) - np.pi/2.)
    if hemisphere == 'upper':
        plunge *= -1.
    if degrees:
        return (bearing*r2d)%360., plunge*r2d
    else:
        return bearing, plunge

def mean_angular_residual(stress_tensor, strikes, dips, rakes):
    return np.mean(np.abs(angular_residual(stress_tensor, strikes, dips, rakes)))

def normal_slip_vectors(strike, dip, rake, direction='inward'):
    """
    Determine the normal and the slip vectors of the
    focal mechanism defined by (strike, dip, rake).
    From Stein and Wysession 2002.
    N.B.: This is the normal of the FOOT WALL and the slip
    of the HANGING WALL w.r.t the foot wall. It means that the
    normal is an inward-pointing normal for the hanging wall,
    and an outward pointing-normal for the foot wall.

    The vectors are in the coordinate system (x1, x2, x3):
    x1: north
    x2: west
    x3: upward
    """
    d2r = np.pi/180.
    strike = strike*d2r
    dip = dip*d2r
    rake = rake*d2r
    n = np.array([-np.sin(dip)*np.sin(strike),
                  -np.sin(dip)*np.cos(strike),
                  np.cos(dip)])
    if direction == 'inward':
        # this formula already gives the inward-pointing
        # normal of the hanging wall
        pass
    elif direction == 'outward':
        n *= -1.
    else:
        print('direction should be either "inward" or "outward"')
        return
    # slip on the hanging wall
    d = np.array([np.cos(rake)*np.cos(strike) + np.sin(rake)*np.cos(dip)*np.sin(strike),
                  -np.cos(rake)*np.sin(strike) + np.sin(rake)*np.cos(dip)*np.cos(strike),
                  np.sin(rake)*np.sin(dip)])
    return n, d

def R_(principal_stresses):
    """
    Compute the shape ratio R=(sig1-sig2)/(sig1-sig3).

    Parameters
    -----------
    pinricpal_stresses: numpy array or list.
        Contains the three eigenvalues of the stress tensor
        ordered such that:
        principal_stresses[0] > principal_stresses[1] > principal_stresses[2]
        with principal_stresses[0] being the most compressional stress.

    Returns
    ---------
    shape_ratio: scalar float.
    """
    return (principal_stresses[0]-principal_stresses[1])\
          /(principal_stresses[0]-principal_stresses[2])

def reduced_stress_tensor(principal_directions, R):
    sig1 = -1.
    sig2 = 2.*R-1.
    sig3 = +1
    Sigma = np.diag(np.array([sig1, sig2, sig3]))
    Sigma /= np.sqrt(np.sum(Sigma**2))
    stress_tensor = np.dot(principal_directions,
                           np.dot(Sigma, principal_directions.T))
    return stress_tensor


def stress_tensor_eigendecomposition(stress_tensor):
    """
    Parameters
    -----------
    stress_tensor: (3, 3) numpy array.
        The stress tensor for which to solve the
        eigenvalue problem.
    Returns
    --------
    principal_stresses: (3,) numpy array.
        The three eigenvalues of the stress tensor, ordered
        from most compressive (sigma1) to least compressive (sigma3).
    principal_directions: (3, 3) numpy array.
        The three eigenvectors of the stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: principal_directions[:, i] 
    """
    try:
        principal_stresses, principal_directions = \
                              np.linalg.eigh(stress_tensor)
    except LinAlgError:
        print(stress_tensor)
        sys.exit()
    #order = np.argsort(principal_stresses)[::-1]
    order = np.argsort(principal_stresses)
    # reorder from most compressive to most extensional
    # with tension positive convention
    # (note: principal_directions is the matrix a column-eigenvectors)
    principal_stresses = principal_stresses[order]
    principal_directions = principal_directions[:, order]
    return principal_stresses, principal_directions

def strike_dip_rake(n, d):
    """
    Invert the relationships between strike/dio/rake
    and normal (n) and slip (d) vectors found in Stein.
    """
    r2d = 180./np.pi
    # ----------------
    # dip is straightforward:
    dip = np.arccos(n[2])
    sin_dip = np.sin(dip)
    if sin_dip != 0.:
        # ----------------
        # strike is more complicated because it spans 0-360 degrees
        sin_strike = -n[0]/sin_dip
        cos_strike = -n[1]/sin_dip
        strike = np.arctan2(sin_strike, cos_strike)
        # ---------------
        # rake is even more complicated
        sin_rake = d[2]/sin_dip
        cos_rake = (d[0] - sin_rake*np.cos(dip)*sin_strike)/cos_strike
        rake = np.arctan2(sin_rake, cos_rake)
    else:
        print('Dip is zero! The strike and rake cannot be determined')
        # the solution is ill-defined, we can only
        # determine rake - strike
        cos_rake_m_strike = d[0]
        sin_rake_m_strike = d[1]
        rake_m_strike = np.arctan2(sin_rake_m_strike, cos_rake_m_strike)
        # fix arbitrarily the rake to zero
        rake = 0.
        strike = -rake_m_strike
    return (strike*r2d)%360., dip*r2d, (rake*r2d)%360.

def shear_slip_angle_difference(stress_tensor, strike, dip, rake):
    """
    Return the angle difference between the slip vector
    from the focal mechanism solution and the shear traction
    on the fault determined from the inverted stress tensor.
    Given that the stress inversion is made under the Wallace-Bott
    assumption, shear stress on the fault is parallel to slip, then
    this angle difference is a measure of misfit.
    """
    # first, get the normal and slip vectors corresponding
    # to (strike, dip, rake)
    n, d = normal_slip_vectors(strike, dip, rake, direction='inward')
    n = n.reshape(1, -1) # make sure it's a row vector
    # second, compute the shear stress on the fault
    traction, normal_traction, shear_traction = compute_traction(
            stress_tensor, n)
    shear_dir = shear_traction/np.sqrt(np.sum(shear_traction**2))
    # the angle difference is the Arccos(dot product)
    angle = np.arccos(np.sum(d.squeeze()*shear_dir.squeeze()))
    # return the result in degrees
    return angle*180./np.pi
