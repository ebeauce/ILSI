import numpy as np
from ILSI import utils_stress

def random_data_HH2001(N, stress_tensor=None):
    """
    Following Hardebeck and Hauksson 2001: The fault plane orientations
    are drawn from a uniform distribution, with the possibility of
    prefentially retaining planes with high shear/normal.
    """
    # initialize the vectors of strikes/dips/rakes
    strikes = np.zeros(N, dtype=np.float32)
    dips = np.zeros(N, dtype=np.float32)
    if stress_tensor is not None:
        p_stress, p_dir = utils_stress.stress_tensor_eigendecomposition(
                stress_tensor)
        # we assume that deviatoric stress is 0.65 the isotropic stress
        dev_to_iso = 0.65
        dev = (p_stress.max() - p_stress.min())/2.
        target_iso = -(1./dev_to_iso * dev) # compression is negative
        iso = (p_stress.max() + p_stress.min())/2.
        iso_corr = target_iso - iso
        p_stress += iso_corr
        stress_tensor = np.dot(p_dir, np.dot(np.diag(p_stress), p_dir.T))
        count = 0
        while count < N:
            strike = 360.*np.random.random()
            dip = 90.*np.random.random()
            # compute the fault normal:
            n, _ = utils_stress.normal_slip_vectors(strike, dip, np.zeros_like(strike))
            # compute normal and shear stress
            T, Tn, Ts = utils_stress.compute_traction(stress_tensor, n[np.newaxis, :])
            ratio = np.linalg.norm(Ts, 2)/np.linalg.norm(Tn, 2)
            if np.random.random() < ratio/dev_to_iso:
                strikes[count] = strike
                dips[count] = dip
                count += 1
    else:
        strikes = 360.*np.random.uniform(size=N)
        dips = 90.*np.random.uniform(size=N)
    return strikes, dips

def random_data_failure(N, stress_tensor, friction_coefficient, min_instability,
                        max_acceptance_probability=0.5, random_friction=0.,
                        naive_uniform=True, plot_density=False):
    """
    """
    np.random.seed(0)
    # get principal faults and compute instability of most unstable fault
    n1, n2 = utils_stress.principal_faults(stress_tensor, friction_coefficient)
    T1, Tn1, Ts1 = utils_stress.compute_traction(stress_tensor, n1.T)
    normal_stress1 = np.sum(Tn1*n1.squeeze())
    shear_stress1 = np.linalg.norm(Ts1, 2)
    p_sig, p_dir = utils_stress.stress_tensor_eigendecomposition(stress_tensor)
    R = utils_stress.R_(p_sig)
    # initialize the vectors of strikes/dips/rakes
    strikes = np.zeros(N, dtype=np.float32)
    dips = np.zeros(N, dtype=np.float32)
    fault_normals = np.zeros((N, 3), dtype=np.float32)
    count = 0
    while count < N:
        strike = 360.*np.random.random()
        dip = 90.*np.random.random()
        # compute the fault normal:
        n, s = utils_stress.normal_slip_vectors(strike, dip, np.zeros_like(strike))
        # compute normal and shear stress
        T, Tn, Ts = utils_stress.compute_traction(stress_tensor, n[np.newaxis, :])
        normal_stress = np.sum(Tn*n)
        shear_stress = np.linalg.norm(Ts, 2)
        # add a random component to the coefficient of friction
        friction_ = friction_coefficient + random_friction*(np.random.random()-0.5)
        # compute the value for cohesion so that the maximum fault
        # instability is the user-prescribed min_instability
        Ic = shear_stress1 - friction_*(p_sig[0] - normal_stress1)
        # compute the fault instability:
        I = (shear_stress - friction_*(p_sig[0] - normal_stress))/Ic
        acceptance_p = I - (1. - max_acceptance_probability)
        if naive_uniform:
            criterion = (I > min_instability or np.random.random() < acceptance_p)
        else:
            criterion = ((I > min_instability and 
                          (np.random.random() < (I-min_instability)/(1.-min_instability)))
                         or np.random.random() < acceptance_p)
        if criterion:
            strikes[count] = strike
            dips[count] = dip
            fault_normals[count, :] = n
            count += 1
    # compare the proximity of each fault plane to the two principal faults
    # to determine in which half of the Mohr space they fall
    # take the absolute value of the inner product because we are truly
    # looking for an angle between two lines rather than two vectors
    angle1 = np.abs(np.arccos(np.abs(np.dot(fault_normals, n1)).squeeze()))
    angle2 = np.abs(np.arccos(np.abs(np.dot(fault_normals, n2)).squeeze()))
    half = np.ones(N, dtype=np.float32)
    half[angle2 < angle1] = -1.
    fig = plot_dataset_Mohr(
            stress_tensor, strikes, dips, shear_sign=half, title='_Mohr',
            plot_density=plot_density)
    ax = fig.get_axes()[0]
    normal_stresses = np.linspace(-0.50, 0.75, 100)
    Ic = shear_stress1 - friction_coefficient*(p_sig[0] - normal_stress1)
    cohesion = Ic*min_instability + friction_coefficient*p_sig[0]
    Mohr_crit = cohesion - friction_coefficient*normal_stresses
    ax.plot(normal_stresses, Mohr_crit, color='C3')
    ax.plot(normal_stresses, -1.*Mohr_crit, color='C3')
    ax.plot(normal_stress1, shear_stress1, marker='o', color='k', markersize=10)
    ax.plot(normal_stress1, -1.*shear_stress1, marker='o', color='k', markersize=10)
    return strikes, dips, fig

def plot_dataset_Mohr(stress_tensor, strikes, dips,
                      shear_sign=None, title=None,
                      plot_density=False, ax=None):
    import matplotlib.pyplot as plt
    # eigendecomposition
    p_stress, p_dir = utils_stress.stress_tensor_eigendecomposition(
            stress_tensor)
    # plot Mohr circle
    if ax is None:
        fig, ax = plt.subplots(num=f'synthetic_dataset{title}', figsize=(18, 9))
    else:
        fig = ax.get_figure()
    sig1, sig2, sig3 = p_stress
    for s1, s2 in [[sig1, sig3], [sig1, sig2], [sig2, sig3]]:
        center_x = (s1+s2)/2.
        radius = (s2-s1)/2.
        circle = plt.Circle((center_x, 0.), radius, color='k', fill=False)
        ax.add_patch(circle)
    normals, slips = utils_stress.normal_slip_vectors(strikes, dips, np.zeros_like(strikes))
    normals, slips = normals.T, slips.T
    traction, normal_traction, shear_traction = utils_stress.compute_traction(
            stress_tensor, normals)
    shear_mag = np.sqrt(np.sum(shear_traction**2, axis=-1))
    if shear_sign is not None:
        shear_mag *= shear_sign
    normal_comp = np.sum(normal_traction*normals, axis=-1)
    for i, s in enumerate([sig1, sig2, sig3]):
        ax.text(s-0.10, -0.04, r'$\sigma_{{{:d}}}$'.format(i+1))
    if plot_density:
        import seaborn as sns
        sns.kdeplot(x=normal_comp, y=shear_mag, ax=ax, fill=True, cmap='inferno', bw_adjust=0.4)
    else:
        ax.scatter(normal_comp, shear_mag)
    ax.set_xlim(-1., +1.)
    ax.set_ylim(-1., +1.)
    ax.set_aspect('equal')
    ax.grid()
    ax.axhline(0., color='k', lw=1.0)
    ax.set_ylabel(r'Shear stress $\tau$')
    ax.set_xlabel(r'Normal stress $\sigma$')
    #ax.legend(loc='lower left', handlelength=0.5)
    return fig

def plot_dataset_PT(strikes, dips, rakes, figname=None, stress_tensor=None):
    import mplstereonet as mpl
    import matplotlib.pyplot as plt
    # compute all normal and slip vectors
    n, s = utils_stress.normal_slip_vectors(strikes, dips, rakes)
    # compute the bearing and plunge of each P/T vector
    p_or = np.zeros((len(strikes), 2), dtype=np.float32)
    t_or = np.zeros((len(strikes), 2), dtype=np.float32)
    for i in range(len(strikes)):
        p, t, b = utils_stress.p_t_b_axes(n[:, i], s[:, i])
        p_or[i, :] = utils_stress.get_bearing_plunge(p)
        t_or[i, :] = utils_stress.get_bearing_plunge(t)
    # plot
    fig = plt.figure(f'PT_axes{figname}', figsize=(18, 9))
    ax = fig.add_subplot(111, projection='stereonet')
    ax.line(t_or[:, 1], t_or[:, 0], marker='v', color='C0')
    ax.line(p_or[:, 1], p_or[:, 0], marker='o', color='C3')
    ax.line(t_or[0, 1], t_or[0, 0], marker='v', color='C0', label='T-axis')
    ax.line(p_or[0, 1], p_or[0, 0], marker='o', color='C3', label='P-axis')
    if stress_tensor is not None:
        # plot the principal stress directions
        p_sig, p_dir = utils_stress.stress_tensor_eigendecomposition(
                stress_tensor)
        colors = ['C3', 'C2', 'C0']
        markers = ['o', 's', 'v']
        for i in range(3):
            # i-th principal direction
            bearing, plunge = utils_stress.get_bearing_plunge(p_dir[:, i])
            ax.line(plunge, bearing, marker=markers[i], color=colors[i],
                    markeredgecolor='k',
                    markersize=15, label=r'$\sigma_{{{:d}}}$'.format(i+1))
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0.90, 0.92)) 
    return fig
