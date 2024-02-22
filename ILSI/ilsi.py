import sys

import numpy as np
import warnings

from functools import partial
from numpy.linalg import LinAlgError
from . import utils_stress

# from time import time as give_time


def forward_model(n_):
    """
    Build the forward modeling matrix ``G`` given a collection
    of fault normals.

    Parameters
    ------------
    n_: (n_earthquakes, 3) numpy.ndarray
        The i-th row n_ are the components of the i-th
        fault normal in the (north, west, south) coordinate
        system.

    Returns
    ---------
    G: (3 x n_earthquakes, 5) numpy.ndarray
        The forward modeling matrix giving the slip (shear stress)
        directions on the faults characterized by `n_`, given the 5
        elements of the deviatoric stress tensor.
    """
    n_earthquakes = n_.shape[0]
    G = np.zeros((n_earthquakes * 3, 5), dtype=np.float32)
    for i in range(n_earthquakes):
        ii = i * 3
        n1, n2, n3 = n_[i, :]
        G[ii + 0, 0] = n1 + n1 * n3**2 - n1**3
        G[ii + 0, 1] = n2 - 2.0 * n2 * n1**2
        G[ii + 0, 2] = n3 - 2.0 * n3 * n1**2
        G[ii + 0, 3] = n1 * n3**2 - n1 * n2**2
        G[ii + 0, 4] = -2.0 * n1 * n2 * n3
        G[ii + 1, 0] = n2 * n3**2 - n2 * n1**2
        G[ii + 1, 1] = n1 - 2.0 * n1 * n2**2
        G[ii + 1, 2] = -2.0 * n1 * n2 * n3
        G[ii + 1, 3] = n2 + n2 * n3**2 - n2**3
        G[ii + 1, 4] = n3 - 2.0 * n3 * n2**2
        G[ii + 2, 0] = n3**3 - n3 - n3 * n1**2
        G[ii + 2, 1] = -2.0 * n1 * n2 * n3
        G[ii + 2, 2] = n1 - 2.0 * n1 * n3**2
        G[ii + 2, 3] = n3**3 - n3 - n3 * n2**2
        G[ii + 2, 4] = n2 - 2.0 * n2 * n3**2
    return G

def _check_apriori_covariances(dim_D, dim_M, C_d=None, C_d_inv=None, C_m=None, C_m_inv=None):
    """Replaces None with default values if needed.
    """
    if C_d is None:
        C_d = np.zeros((dim_D, dim_D), dtype=np.float32)
        C_d_inv = np.identity(dim_D, dtype=np.float32)
    elif C_d_inv is None and inversion_space == "model_space":
        try:
            C_d_inv = np.linalg.inv(C_d)
        except LinAlgError:
            print("Cannot invert data covariance matrix:")
            print(C_d)
            sys.exit()
    if C_m is None:
        C_m = np.identity(dim_M, dtype=np.float32)
        C_m_inv = np.zeros_like(C_m)
    elif C_m_inv is None and inversion_space == "model_space":
        try:
            C_m_inv = np.linalg.inv(C_m)
        except LinAlgError:
            print("Cannot invert model covariance matrix:")
            print(C_m)
            sys.exit()
    return C_d, C_d_inv, C_m, C_m_inv

def Tarantola_Valette(
    G,
    data,
    C_d=None,
    C_d_inv=None,
    C_m=None,
    C_m_inv=None,
    m_prior=None,
    inversion_space="model_space",
):
    """
    Returns Tarantola's and Valette's least square solution for
    a given linear operator `G` and observation vector `data`. If the
    covariance matrices of the observations and of the model
    parameters are not known, we assume them to be identity. The
    inversion can be performed either in the data space or in
    the model space.

    Parameters
    -----------
    G: (n, m) numpy.ndarray
        The linear operator projecting elements of the model
        space m onto the data space: d = G.m
        n is the dimension of the data space,
        m is the dimension of the model space.
    data: (3k,) or (3k, 1) or (k, 3) numpy.ndarray
        Vector of observations. k is the number of focal mechanisms. `data`
        is reshaped to (n=3k, 1) before the inversion.
    C_d: (n, n) numpy.ndarray, default to None
        Covariance matrix of the observations. It quantifies
        the errors in the observations and propagates them
        in the inversion to give more weight to the observations
        with low errors. If None, then `C_d` is filled with zeros
        (assume no error in data).
    C_m: (m, m) numpy.ndarray, default to None
        Covariance matrix of the model parameters. It quantifies
        the errors in the model parameters and propagates them
        in the inversion to determine the range of acceptable
        model parameters for a given set of observations.
        If None, then `C_m` is identity.
    m_prior: (m,) or (m, 1) numpy.ndarray, default to None
        If one already has a rough estimate of what the model
        parameters are, then m_prior should be filled with this estimate.
        If None, `m_prior` is set to zero.

    Returns
    ---------
    m_inv: (m, 1) numpy.ndarray
        The inverted model parameters.
    C_m_posterior: (5, 5) numpy.ndarray
        Posterior covariance of the model parameter distribution.
    C_d_posterior: (3 x n_earthquakes, 3 x n_earthquakes) numpy.ndarray
        Posterior covariance of the data distribution.
    """
    # t_start = give_time()
    dim_D = G.shape[0]
    dim_M = G.shape[1]
    C_d, C_d_inv, C_m, C_m_inv = _check_apriori_covariances(
            dim_D, dim_M, C_d=C_d, C_d_inv=C_d_inv, C_m=C_m, C_m_inv=C_m_inv
            )
    if m_prior is None:
        m_prior = np.zeros((dim_M, 1), dtype=np.float32)
    # make sure data is a column vector
    data = data.reshape(-1, 1)
    if inversion_space == "data_space":
        # perform the inversion in the data space
        # pre-compute recurrent terms:
        Cm_Gt = C_m.dot(G.T)
        inv = np.linalg.inv(G.dot(Cm_Gt) + C_d)
        Cm_Gt_inv = Cm_Gt.dot(inv)
        m_inv = m_prior + np.dot(Cm_Gt_inv, data - np.dot(G, m_prior))
        C_m_posterior = C_m - (Cm_Gt_inv.dot(G)).dot(C_m)
    elif inversion_space == "model_space":
        # perform the inversion in the model space
        # pre-compute recurrent terms
        Gt_Cdinv = G.T.dot(C_d_inv)
        try:
            inv = np.linalg.inv(Gt_Cdinv.dot(G) + C_m_inv)
        except LinAlgError:
            print("Could not solve the inverse problem in the model space.")
            print("Forward modelling matrix:", G)
            print("Inverse data cov matrix:", C_d_inv)
            print("Inverse model cov matrix:", C_m_inv)
            print(np.dot(Gt_Cdinv, G))
            sys.exit()
        m_inv = m_prior + (inv.dot(Gt_Cdinv)).dot(data - G.dot(m_prior))
        C_m_posterior = inv.copy()
    else:
        print('inversion_spce should either be "model_space" ' 'or "data_space"')
        return
    C_d_posterior = (G.dot(C_m_posterior)).dot(G.T)
    # t_end = give_time()
    # print('{:.3f}sec on Tarantola'.format(t_end-t_start))
    return m_inv, C_m_posterior, C_d_posterior


def iterative_linear_si(
    strikes,
    dips,
    rakes,
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    Tarantola_kwargs={},
    return_eigen=True,
    return_stats=False,
):
    """
    Iterative stress inversion described in Beauce et al. 2022.

    This method assumes:
        - The tectonic stress field is uniform.
        - Wallace-Bott hypothesis: The slip vector points in the same
          direction as shear stress on the fault.
    The parameters we invert for are the directions of the three
    principal stresses and the shape ratio. Because this inversion does not
    aim at infering the absolute stress values, we only consider the
    deviatoric stress tensor, therefore Trace(sigma) = 0. Furthermore, we cannot
    determine the norm of the stress tensor, therefore sum sigma**2 = 1.
    Each iteration of this inversion scheme is a linear inversion.
    N.B.: This routine is written assuming outward footwall normals and slip
    vectors of the hanging wall w.r.t. the footwall. Therefore, the stress
    tensor sign convention is compression negative.

    Parameters
    -----------
    strikes: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    Tarantola_kwargs: Dictionary, default to {}:
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    return_eigen: boolean, default to True
        If True, returns the eigendecomposition of the inverted
        stress tensor in addition to returning the stress tensor.
    return_stats: boolean, default to True
        If True, the posterior data and model parameter distributions
        estimated from the Tarantola and Valette formula
        (cf. Tarantola_Valette routine).

    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["stress_tensor"]: (3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["principal_stresses"]: (3,) numpy.ndarray, optional
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
            Returned if `return_eigen` is True.
        - output["principal_directions"]: (3, 3) numpy.ndarray, optional
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i is
            given by: `principal_directions[:, i]`.
            Returned if `return_eigen` is True.
        - output["C_m_posterior"]: (5, 5) numpy.ndarray, optional
            Posterior covariance of the model parameter distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["C_d_posterior"]: (3 x n_earthquakes, 3 x n_earthquakes) numpy.ndarray, optional
            Posterior covariance of the data distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["resolution_operator"]: (3, 3) numpy.ndarray
            The resolution operator assuming that the shear tractions are all
            perfectly constant or that they are all perfectly estimated.
    """
    # t_start = give_time()
    # First, convert the strike/dip/rake into slip and normal vectors.
    n_earthquakes = len(strikes)
    n_, d_ = utils_stress.normal_slip_vectors(strikes, dips, rakes, direction="inward")
    n_, d_ = n_.T, d_.T
    # Next, define the matrix that relates the stress tensor
    # to the observed slip vectors, given the fault geometries
    # characterized by the normal vectors.
    # For each earthquake, an (3 x 5) matrix relates the 5 independent
    # stress tensor components to the 3 slip vector components.
    G = forward_model(n_)
    if Tarantola_kwargs is not None:
        # -----------------------------------------
        # copy Tarantola_kwargs because all modifications are made in-place
        Tarantola_kwargs = Tarantola_kwargs.copy()
        method = "tarantola"
    else:
        method = "moore_penrose"
    # initialize shear magnitudes
    if method == "tarantola" and "m_prior" in Tarantola_kwargs:
        shear = np.sqrt(
            np.sum(
                (G @ Tarantola_kwargs["m_prior"].astype("float32")).reshape(
                    n_earthquakes, 3
                )
                ** 2,
                axis=-1,
            )
        )
    else:
        shear = np.ones(n_earthquakes, dtype=np.float32)
    for j in range(max_n_iterations):
        if method == "tarantola":
            sigma, C_m_posterior, C_d_posterior = Tarantola_Valette(
                G, d_ * shear[:, np.newaxis], **Tarantola_kwargs
            )
            sigma = sigma.squeeze()
        elif method == "moore_penrose":
            # We choose any inversion method to invert G:
            G_pinv = np.linalg.pinv(G)
            # Given how we defined G, the stress tensor components
            # we get are in this order:
            # sigma_11, sigma_12, sigma_13, sigma_22, sigma_23
            sigma = np.dot(G_pinv, d_.reshape(-1, 1)).squeeze()
        # normalize the stress tensor to make sure the units of
        # shear does not explode or vanish (it can behave like a
        # geometrical series), this normalization gives the reduced
        # stress tensor (up to a multiplicative constant)
        full_stress_tensor = np.array(
            [
                [sigma[0], sigma[1], sigma[2]],
                [sigma[1], sigma[3], sigma[4]],
                [sigma[2], sigma[4], -sigma[0] - sigma[3]],
            ]
        )
        # normalize by trace of squared matrix
        norm = np.sqrt(np.sum(np.diag(full_stress_tensor)**2))
        norm = 1 if norm == 0.0 else norm
        sigma /= norm
        if method == "tarantola":
            Tarantola_kwargs["m_prior"] = sigma.reshape(5, 1).copy()
        # Note: From Tarantola's book: in an iterative non-linear
        # inversion, he does not input the posterior distribution from
        # previous iteration to the next iteration. Doing so leads to
        # vanishing or exploding covariance matrices!
        # -----------------------------
        # compute shear magnitudes
        shear0 = shear.copy()
        predicted_shear = (G @ sigma).reshape(n_earthquakes, 3)
        shear = np.sqrt(np.sum(predicted_shear**2, axis=-1))
        shear_update = np.sqrt(np.mean((shear - shear0) ** 2))
        # print('Shear stress update: {:.3e}'.format(shear_update))
        if shear_update < shear_update_atol:
            # convergence has been reached, according to the
            # user-prescribed criterion
            # print('Stop at iteration {:d}! (shear update: {:.3e})'.format(j, shear_update))
            break
    # `sigma` is already normalized 
    sigma = sigma.squeeze()
    # build full stress tensor
    full_stress_tensor = np.array(
        [
            [sigma[0], sigma[1], sigma[2]],
            [sigma[1], sigma[3], sigma[4]],
            [sigma[2], sigma[4], -sigma[0] - sigma[3]],
        ]
    )
    # use `norm` from last iteration and normalize cov matrix
    C_m_posterior /= norm**2
    # return output in dictionary
    output = {}
    output["stress_tensor"] = full_stress_tensor
    output["predicted_shear_stress"] = predicted_shear
    if return_eigen:
        # solve the eigenvalue problem
        (
            principal_stresses,
            principal_directions,
        ) = utils_stress.stress_tensor_eigendecomposition(full_stress_tensor)
        output["shear_stress_magnitudes"] = shear
        output["principal_stresses"] = principal_stresses
        output["principal_directions"] = principal_directions
    if return_stats:
        if method == "tarantola":
            output["C_d_posterior"] = C_d_posterior
            output["C_m_posterior"] = C_m_posterior
            _, C_d_inv, _, C_m_inv = _check_apriori_covariances(
                    *G.shape,
                    C_d=Tarantola_kwargs.get("C_d", None),
                    C_d_inv=Tarantola_kwargs.get("C_d_inv", None),
                    C_m=Tarantola_kwargs.get("C_m", None),
                    C_m_inv=Tarantola_kwargs.get("C_m_inv", None),
                    )
            output["resolution_operator"] = utils_stress.resolution_operator(
                    G, C_d_inv, C_m_inv
                    )
        else:
            warnings.warn(
                    "return_stats=True only works with the Tarantola-Valette "
                    "method (see Tarantola_kwargs), returning dummy outputs"
                    )
            output["C_d_posterior"] = np.diag(np.ones(3 * n_earthquakes, dtype=np.float32))
            output["C_m_posterior"] = np.ones((5, 5), dtype=np.float32)
            output["resolution_operator"] = np.ones((3, 3), dtype=np.float32)
    # t_end = give_time()
    # print('iterative_linear_si finished in {:.2f}sec'.format(t_end-t_start))
    return output


def Michael1984_inversion(
    strikes, dips, rakes, Tarantola_kwargs={}, return_eigen=True, return_stats=False
):
    """
    Linear inversion described in Michael 1984.

    This method assumes:
        - The tectonic stress field is uniform.
        - Wallace-Bott hypothesis: The slip vector points in the same
          direction as shear stress on the fault.
        - The resolved shear stress magnitude is constant on
          all faults.
    The parameters we invert for are the directions of the three
    principal stresses and the shape ratio. Because this inversion does not
    aim at infering the absolute stress values, we only consider the
    deviatoric stress tensor, therefore Trace(sigma) = 0. Furthermore, we cannot
    determine the norm of the stress tensor, therefore sum sigma**2 = 1.
    Each iteration of this inversion scheme is a linear inversion.
    N.B.: This routine is written assuming outward footwall normals and slip
    vectors of the hanging wall w.r.t. the footwall. Therefore, the stress
    tensor sign convention is compression negative.

    Parameters
    -----------
    strikes: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    return_eigen: boolean, default to True
        If True, returns the eigendecomposition of the inverted
        stress tensor in addition to returning the stress tensor.
    return_stats: boolean, default to True
        If True, the posterior data and model parameter distributions
        estimated from the Tarantola and Valette formula
        (cf. Tarantola_Valette routine).

    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["stress_tensor"]: (3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["principal_stresses"]: (3,) numpy.ndarray, optional
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
            Returned if `return_eigen` is True.
        - output["predicted_shear_stress"]: (n_earthquakes, 3) numpy.ndarray
            The shear tractions resolved on the fault planes described by `strikes`,
            `dips` and `rakes` computed with the inverted stress tensor.
        - output["principal_directions"]: (3, 3) numpy.ndarray, optional
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i is
            given by: `principal_directions[:, i]`.
            Returned if `return_eigen` is True.
        - output["C_m_posterior"]: (5, 5) numpy.ndarray, optional
            Posterior covariance of the model parameter distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["C_d_posterior"]: (3 x n_earthquakes, 3 x n_earthquakes) numpy.ndarray, optional
            Posterior covariance of the data distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["resolution_operator"]: (3, 3) numpy.ndarray
            The resolution operator assuming that the shear tractions are all
            perfectly constant or that they are all perfectly estimated.
    """
    # First, convert the strike/dip/rake into slip and normal vectors.
    n_earthquakes = len(strikes)
    n_ = np.zeros((n_earthquakes, 3), dtype=np.float32)  # normal vectors
    d_ = np.zeros((n_earthquakes, 3), dtype=np.float32)  # slip vectors
    for i in range(n_earthquakes):
        n_[i, :], d_[i, :] = utils_stress.normal_slip_vectors(
            strikes[i], dips[i], rakes[i], direction="inward"
        )
    # Next, define the matrix that relates the stress tensor
    # to the observed slip vectors, given the fault geometries
    # characterized by the normal vectors.
    # For each earthquake, an (3 x 5) matrix relates the 5 independent
    # stress tensor components to the 3 slip vector components.
    G = forward_model(n_)
    if Tarantola_kwargs is not None:
        sigma, C_m_posterior, C_d_posterior = Tarantola_Valette(
            G, d_, **Tarantola_kwargs
        )
        sigma = sigma.squeeze()
        method = "tarantola"
    else:
        # We choose any inversion method to invert G:
        G_pinv = np.linalg.pinv(G)
        # Given how we defined G, the stress tensor components
        # we get are in this order:
        # sigma_11, sigma_12, sigma_13, sigma_22, sigma_23
        sigma = np.dot(G_pinv, d_.reshape(-1, 1)).squeeze()
        method = "moore_penrose"
    full_stress_tensor = np.array(
        [
            [sigma[0], sigma[1], sigma[2]],
            [sigma[1], sigma[3], sigma[4]],
            [sigma[2], sigma[4], -sigma[0] - sigma[3]],
        ]
    )
    norm = np.sqrt(np.sum(full_stress_tensor**2))
    norm = 1 if norm == 0.0 else norm
    full_stress_tensor /= norm
    # return output in dictionary
    output = {}
    output["stress_tensor"] = full_stress_tensor
    output["predicted_shear_stress"] = (G @ sigma).reshape(n_earthquakes, 3)
    if return_eigen:
        # solve the eigenvalue problem
        (
            principal_stresses,
            principal_directions,
        ) = utils_stress.stress_tensor_eigendecomposition(full_stress_tensor)
        output["principal_stresses"] = principal_stresses
        output["principal_directions"] = principal_directions
    if return_stats:
        if method == "tarantola":
            output["C_d_posterior"] = C_d_posterior
            output["C_m_posterior"] = C_m_posterior
            _, C_d_inv, _, C_m_inv = _check_apriori_covariances(
                    *G.shape,
                    C_d=Tarantola_kwargs.get("C_d", None),
                    C_d_inv=Tarantola_kwargs.get("C_d_inv", None),
                    C_m=Tarantola_kwargs.get("C_m", None),
                    C_m_inv=Tarantola_kwargs.get("C_m_inv", None),
                    )
            output["resolution_operator"] = utils_stress.resolution_operator(
                    G, C_d_inv, C_m_inv
                    )
        else:
            warnings.warn(
                    "return_stats=True only works with the Tarantola-Valette "
                    "method (see Tarantola_kwargs), returning dummy outputs"
                    )
            output["C_d_posterior"] = np.diag(np.ones(3 * n_earthquakes, dtype=np.float32))
            output["C_m_posterior"] = np.ones((5, 5), dtype=np.float32)
            output["resolution_operator"] = np.ones((3, 3), dtype=np.float32)
    return output


# ---------------------------------------------------
#
#   Routines for inversion without instability parameter
#
# ---------------------------------------------------


def inversion_one_set(
    strikes,
    dips,
    rakes,
    n_random_selections=20,
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    variable_shear=True,
    Tarantola_kwargs={},
    return_eigen=True,
    return_stats=False,
    input_fault_planes=False,
):
    """
    Invert one set of focal mechanisms without seeking which nodal planes
    are more likely to be the fault planes.

    Parameters
    -----------
    strikes: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    n_random_selections: integer, default to 20
        Number of random selections of subsets of nodal planes on
        which the stress inversion is run. The final stress tensor
        is averaged over the n_random_selections solutions.
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    variable_shear: boolean, default to True
        If True, use the iterative linear method described in
        Beauce et al. 2022, else use the classic linear method
        due to Michael 1984.
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    return_stats: boolean, default to True
        If True, the posterior data and model parameter distributions
        estimated from the Tarantola and Valette formula
        (cf. Tarantola_Valette routine).
    input_fault_planes : boolean, optional
        If True, `strikes, `dips` and `rakes` are considered to be
        the fault planes and no auxiliary plane will be computed.

    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["stress_tensor"]: (3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["principal_stresses"]: (3,) numpy.ndarray, optional
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
            Returned if `return_eigen` is True.
        - output["principal_directions"]: (3, 3) numpy.ndarray, optional
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i is
            given by: `principal_directions[:, i]`.
            Returned if `return_eigen` is True.
        - output["C_m_posterior"]: (5, 5) numpy.ndarray, optional
            Posterior covariance of the model parameter distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["C_d_posterior"]: (3 x n_earthquakes, 3 x n_earthquakes) numpy.ndarray, optional
            Posterior covariance of the data distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["resolution_operator"]: (3, 3) numpy.ndarray
            The resolution operator assuming that the shear tractions are all
            perfectly constant or that they are all perfectly estimated.
    """
    # compute auxiliary planes
    strikes_1, dips_1, rakes_1 = strikes, dips, rakes
    if input_fault_planes:
        strikes_2, dips_2, rakes_2 = strikes_1, dips_1, rakes_1
    else:
        strikes_2, dips_2, rakes_2 = np.asarray(
            list(map(utils_stress.aux_plane, strikes, dips, rakes))
        ).T
    # define shape variable
    n_earthquakes = len(strikes)
    # define flat arrays
    strikes = np.hstack((strikes_1.reshape(-1, 1), strikes_2.reshape(-1, 1))).flatten()
    dips = np.hstack((dips_1.reshape(-1, 1), dips_2.reshape(-1, 1))).flatten()
    rakes = np.hstack((rakes_1.reshape(-1, 1), rakes_2.reshape(-1, 1))).flatten()
    # initialize the average stress tensor array
    avg_stress_tensor = np.zeros((3, 3), dtype=np.float32)
    # initialize posterior covariance matrices
    avg_C_m_posterior = np.zeros((5, 5), dtype=np.float32)
    avg_C_d_posterior = np.zeros(
        (3 * n_earthquakes, 3 * n_earthquakes), dtype=np.float32
    )
    # initialize resolution operator
    avg_resolution_operator = np.zeros((5, 5), dtype=np.float32)
    # randomly select subsets of nodal planes and invert for the stress tensor
    for n in range(n_random_selections):
        nodal_planes = np.random.randint(0, 2, size=n_earthquakes)
        flat_indexes = np.int32(np.arange(n_earthquakes) * 2 + nodal_planes)
        selected_strikes = strikes[flat_indexes]
        selected_dips = dips[flat_indexes]
        selected_rakes = rakes[flat_indexes]
        # invert this subset of nodal planes
        if variable_shear:
            # invert for both the stress tensor and values
            # of (normalized) resolved shear stress magnitude
            output_ = iterative_linear_si(
                selected_strikes,
                selected_dips,
                selected_rakes,
                max_n_iterations=max_n_iterations,
                shear_update_atol=shear_update_atol,
                Tarantola_kwargs=Tarantola_kwargs,
                return_eigen=False,
                return_stats=True,
            )
        else:
            # invert only for the stress tensor, assuming
            # constant shear stress on all faults
            output_ = Michael1984_inversion(
                selected_strikes,
                selected_dips,
                selected_rakes,
                Tarantola_kwargs=Tarantola_kwargs,
                return_eigen=False,
                return_stats=True,
            )
        # add them to the average
        avg_stress_tensor += output_["stress_tensor"]
        avg_C_m_posterior += output_["C_m_posterior"]
        avg_C_d_posterior += output_["C_d_posterior"]
        avg_resolution_operator += output_["resolution_operator"]
    avg_stress_tensor /= float(n_random_selections)
    avg_C_m_posterior /= float(n_random_selections)
    avg_C_d_posterior /= float(n_random_selections)
    avg_resolution_operator /= float(n_random_selections)
    norm = np.sqrt(np.sum(avg_stress_tensor**2))
    norm = 1 if norm == 0.0 else norm
    avg_stress_tensor /= norm
    # return output in dictionary
    output = {}
    output["stress_tensor"] = avg_stress_tensor
    if return_eigen:
        # solve the eigenvalue problem
        (
            principal_stresses,
            principal_directions,
        ) = utils_stress.stress_tensor_eigendecomposition(avg_stress_tensor)
        output["principal_stresses"] = principal_stresses
        output["principal_directions"] = principal_directions
    if return_stats:
        if Tarantola_kwargs is not None:
            output["C_d_posterior"] = avg_C_d_posterior
            output["C_m_posterior"] = avg_C_m_posterior
            output["resolution_operator"] = avg_resolution_operator
        else:
            warnings.warn(
                    "return_stats=True only works with the Tarantola-Valette "
                    "method (see Tarantola_kwargs), returning dummy outputs"
                    )
            output["C_d_posterior"] = np.diag(np.ones(3 * n_earthquakes, dtype=np.float32))
            output["C_m_posterior"] = np.ones((5, 5), dtype=np.float32)
            output["resolution_operator"] = np.ones((3, 3), dtype=np.float32)
    return output


def inversion_jackknife(
    jack_strikes,
    jack_dips,
    jack_rakes,
    n_random_selections=1,
    n_resamplings=100,
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    variable_shear=True,
    Tarantola_kwargs={},
    bootstrap_events=False,
    input_fault_planes=False,
):
    """
    This routine was tailored for one of my application, but it can
    be of interest to others. Each earthquake comes with an ensemble
    of focal mechanism solutions that were obtained by resampling the
    set of seismic stations used in the focal mechanism inversion. The
    resampling was done with the delete-k-jackknife method, hence the
    name of the routine. This routine randomly samples focal mechanisms
    from these ensembles and runs the stress inversion. This is a way
    of propagating the focal mechanism uncertainties into the stress
    inversion. In this routine we do not seek which nodal planes are
    more likely to be the fault planes.

    Parameters
    -----------
    jack_strikes: (n_earthquakes, n_jackknifes) numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    jack_dips: (n_earthquakes, n_jackknifes) numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    jack_rakes: (n_earthquakes, n_jackknifes) numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    n_random_selections: integer, default to 20
        Number of random selections of subsets of nodal planes on
        which the stress inversion is run. The final stress tensor
        is averaged over the n_random_selections solutions.
    n_resamplings: integer, default to 100
        Number of times the data set is resampled from the ensembles
        of focal mechanism solutions available for each earthquake.
        n_resamplings stress tensors are returned, allowing to
        estimate uncertainties from the distribution of solutions.
    bootstrap_events: boolean, default to False
        If True, the resampling is also done accross earthquakes,
        following the bootstrapping method.
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    variable_shear: boolean, default to True
        If True, use the iterative linear method described in
        Beauce et al. 2022, else use the classic linear method
        due to Michael 1984.
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    input_fault_planes : boolean, optional
        If True, `strikes, `dips` and `rakes` are considered to be
        the fault planes and no auxiliary plane will be computed.


    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["jack_stress_tensor"]: (n_resamplings, 3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["jack_principal_stresses"]: (n_resamplings, 3) numpy.ndarray
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
        - output["jack_principal_directions"]: (n_resamplings, 3, 3) numpy.ndarray
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i for
            the b-th jackknife replica is given by:
            `jack_principal_directions[b, :, i]`.
    """
    # compute auxiliary planes
    jack_strikes_1, jack_dips_1, jack_rakes_1 = jack_strikes, jack_dips, jack_rakes
    if input_fault_planes:
        jack_strikes_2, jack_dips_2, jack_rakes_2 = jack_strikes_1, jack_dips_1, jack_rakes_1
    else:
        jack_strikes_2, jack_dips_2, jack_rakes_2 = np.asarray(
            list(map(utils_stress.aux_plane, jack_strikes, jack_dips, jack_rakes))
        ).T
    # define shape variables
    n_earthquakes, n_jackknife = jack_strikes_1.shape
    n_planes_per_tp = int(n_jackknife * 2)
    # define flat arrays
    jack_strikes = np.hstack(
        (jack_strikes_1[..., np.newaxis], jack_strikes_2[..., np.newaxis])
    ).flatten()
    jack_dips = np.hstack(
        (jack_dips_1[..., np.newaxis], jack_dips_2[..., np.newaxis])
    ).flatten()
    jack_rakes = np.hstack(
        (jack_rakes_1[..., np.newaxis], jack_rakes_2[..., np.newaxis])
    ).flatten()
    # initialize the average stress tensor arrays
    jack_avg_stress_tensors = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    jack_principal_stresses = np.zeros((n_resamplings, 3), dtype=np.float32)
    jack_principal_directions = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    for b in range(n_resamplings):
        bootstrap_fm = np.random.randint(0, n_jackknife, size=n_earthquakes) * 2
        if bootstrap_events:
            bootstrap_ev = np.random.choice(
                np.arange(n_earthquakes), size=n_earthquakes
            )
        # see inversion_bootstrap for reason for commenting the extra loop
        # for n in range(n_random_selections):
        nodal_planes = np.random.randint(0, 2, size=n_earthquakes)
        if bootstrap_events:
            flat_indexes = np.int32(
                bootstrap_ev * n_planes_per_tp + bootstrap_fm + nodal_planes
            )
        else:
            flat_indexes = np.int32(
                np.arange(n_earthquakes) * n_planes_per_tp + bootstrap_fm + nodal_planes
            )
        selected_strikes = jack_strikes[flat_indexes]
        selected_dips = jack_dips[flat_indexes]
        selected_rakes = jack_rakes[flat_indexes]
        # invert this subset of nodal planes
        if variable_shear:
            # invert for both the stress tensor and values
            # of (normalized) resolved shear stress magnitude
            output_ = iterative_linear_si(
                selected_strikes,
                selected_dips,
                selected_rakes,
                max_n_iterations=max_n_iterations,
                shear_update_atol=shear_update_atol,
                Tarantola_kwargs=Tarantola_kwargs,
                return_eigen=False,
                return_stats=False,
            )
        else:
            # invert only for the stress tensor, assuming
            # constant shear stress on all faults
            output_ = Michael1984_inversion(
                selected_strikes,
                selected_dips,
                selected_rakes,
                Tarantola_kwargs=Tarantola_kwargs,
                return_eigen=False,
                return_stats=False,
            )
        jack_avg_stress_tensors[b, ...] = output_["stress_tensor"]
        (
            jack_principal_stresses[b, ...],
            jack_principal_directions[b, ...],
        ) = utils_stress.stress_tensor_eigendecomposition(
            jack_avg_stress_tensors[b, ...]
        )
    output = {}
    output["jack_stress_tensor"] = jack_avg_stress_tensors
    output["jack_principal_stresses"] = jack_principal_stresses
    output["jack_principal_directions"] = jack_principal_directions
    return output


def inversion_bootstrap(
    strikes,
    dips,
    rakes,
    n_random_selections=1,
    n_resamplings=100,
    variable_shear=True,
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    Tarantola_kwargs={},
    input_fault_planes=False,
):
    """
    Inverts one set of focal mechanisms without seeking which nodal planes
    are more likely to be the fault planes. Performs bootstrap resampling
    of the data set to return an ensemble of solutions.

    Parameters
    -----------
    strikes: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    n_random_selections: integer, default to 5
        Number of random selections of subsets of nodal planes on
        which the stress inversion is run. The final stress tensor
        is averaged over the n_random_selections solutions.
    n_resamplings: integer, default to 100
        Number of times the data set is resampled following
        the bootstrapping method (sampling with replacement).
        n_resamplings stress tensors are returned, allowing to
        estimate uncertainties from the distribution of solutions.
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    variable_shear: boolean, default to True
        If True, use the iterative linear method described in
        Beauce et al. 2022, else use the classic linear method
        due to Michael 1984.
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    input_fault_planes : boolean, optional
        If True, `strikes, `dips` and `rakes` are considered to be
        the fault planes and no auxiliary plane will be computed.


    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["boot_stress_tensor"]: (n_resamplings, 3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["boot_principal_stresses"]: (n_resamplings, 3) numpy.ndarray
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
        - output["boot_principal_directions"]: (n_resamplings, 3, 3) numpy.ndarray
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i for
            the b-th bootstrap replica is given by:
            `boot_principal_directions[b, :, i]`.
    """

    # compute auxiliary planes
    strikes_1, dips_1, rakes_1 = strikes, dips, rakes
    if input_fault_planes:
        strikes_2, dips_2, rakes_2 = strikes_1, dips_1, rakes_1
    else:
        strikes_2, dips_2, rakes_2 = np.asarray(
            list(map(utils_stress.aux_plane, strikes, dips, rakes))
        ).T
    # define shape variables
    n_earthquakes = len(strikes_1)
    n_planes_per_ev = 2
    # initialize the average stress tensor arrays
    boot_avg_stress_tensors = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    boot_principal_stresses = np.zeros((n_resamplings, 3), dtype=np.float32)
    boot_principal_directions = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    # flatten strikes/dips/rakes of planes
    strikes = np.stack((strikes_1, strikes_2), axis=1).flatten()
    dips = np.stack((dips_1, dips_2), axis=1).flatten()
    rakes = np.stack((rakes_1, rakes_2), axis=1).flatten()
    for b in range(n_resamplings):
        if b % 100 == 0:
            print(f"---------- Bootstrapping {b+1}/{n_resamplings} ----------")
        bootstrap_set = np.random.choice(
            np.arange(n_earthquakes), replace=True, size=n_earthquakes
        )
        # I now believe there is no point in adding this extra loop
        # in the bootstrapping method. Averaging only disturbs the
        # uncertainty estimation
        # for n in range(n_random_selections):
        nodal_planes = np.random.randint(0, 2, size=n_earthquakes)
        flat_indexes = np.int32(bootstrap_set * n_planes_per_ev + nodal_planes)
        selected_strikes = strikes[flat_indexes]
        selected_dips = dips[flat_indexes]
        selected_rakes = rakes[flat_indexes]
        # invert this subset of nodal planes
        if variable_shear:
            output_ = iterative_linear_si(
                selected_strikes,
                selected_dips,
                selected_rakes,
                return_eigen=False,
                Tarantola_kwargs=Tarantola_kwargs,
            )
        else:
            output_ = Michael1984_inversion(
                selected_strikes,
                selected_dips,
                selected_rakes,
                return_eigen=False,
                Tarantola_kwargs=Tarantola_kwargs,
            )
        boot_avg_stress_tensors[b, ...] = output_["stress_tensor"]
        (
            boot_principal_stresses[b, ...],
            boot_principal_directions[b, ...],
        ) = utils_stress.stress_tensor_eigendecomposition(
            boot_avg_stress_tensors[b, ...]
        )
    output = {}
    output["boot_stress_tensor"] = boot_avg_stress_tensors
    output["boot_principal_stresses"] = boot_principal_stresses
    output["boot_principal_directions"] = boot_principal_directions
    return output


# ---------------------------------------------------
#
#   Routines for inversion with instability parameter
#
# ---------------------------------------------------


def inversion_one_set_instability(
    strikes,
    dips,
    rakes,
    friction_coefficient=0.6,
    friction_min=0.1,
    friction_max=0.8,
    friction_step=0.05,
    n_stress_iter=10,
    n_random_selections=20,
    stress_tensor_update_atol=1.0e-4,
    Tarantola_kwargs={},
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    n_averaging=1,
    signed_instability=False,
    verbose=True,
    variable_shear=True,
    return_stats=False,
    weighted=False,
    plot=False,
):
    """
    Invert one set of focal mechanisms with the instability parameter
    to seek which nodal planes are more likely to be the fault planes
    (cf. B. Lund and R. Slunga 1999, V. Vavrycuk 2013,2014).
    In general, you can keep the default parameter values.

    Parameters
    -----------
    strikes: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    friction_coefficient: float or None, default to 0.6
        If not None, the inversion is made assuming a friction coefficient
        equal to `friction_coefficient`. If None, the friction coefficient
        is taken as the one that maximizes instability based on a first
        approximation of the stress tensor.
    friction_min: float, default to 0.1
        Lower bound of explored friction values.
    friction_max: float, default to 0.8
        Upper bound of explored friction values.
    friction_step: float, default to 0.05
        Step employed in the grid search of the friction value
        that maximizes the instability parameter.
    n_stress_iter: integer, default to 10
        Maximum number of iterations to seek for the best fault planes.
        See Beauce et al. 2022 for explanations.
    stress_tensor_update_atol: float, default to 1.e-4
        If the RMS difference of the stress tensors between two
        iterations fall below this threshold, convergence has been reached.
    n_random_selections: integer, default to 20
        Number of random selections of subsets of nodal planes on
        which the stress inversion is run. The final stress tensor
        is averaged over the n_random_selections solutions.
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    variable_shear: boolean, default to True
        If True, use the iterative linear method described in
        Beauce et al. 2022, else use the classic linear method
        due to Michael 1984.
    n_averaging: integer, default to 1
        The inversion can be sensitive to initial conditions. To improve
        reproducibility of the results it is good to repeat the inversion
        several times and average the results. Set `n_averaging` to ~5 if
        you can afford the increase in run time.
    signed_instability: boolean, default to False
        If True, the instability parameter ranges from -1 to +1. Negative
        values mean that the predicted and observed slip have opposite
        directions. If False, the instability parameter is the one
        defined in Vavrycuk 2013, 2014.
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    return_stats: boolean, default to True
        If True, the posterior data and model parameter distributions
        estimated from the Tarantola and Valette formula
        (cf. Tarantola_Valette routine).
    weighted: boolean, default to False
        This option is exploratory. If True:
            1) More weight is given to the fault planes that are clearly
               more unstable than their auxiliary counterpart in the
               stress field estimated at iteration t-1
            2) Randomly mixes the set of fault planes at iterations
               t-1 and t, giving larger probability to the planes
               belonging to the set that produced the larger instability.
        This option can be interesting for reaching convergence on
        data sets of bad quality.
    plot: boolean, default to False
        If True, plot the set of nodal planes selected at each iteration,
        and the weight attributed to each of these planes. Can be used
        with `weighted=True` to see if the inversion convergences to a
        well defined set of planes.
    verbose: integer, default to 1
        Level of verbosity.
        0: No print statements.
        1: Print whether the algorithm converged.
        2: Print the stress tensor at the end of each fault plane
           selection iteration.


    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["stress_tensor"]: (3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["friction_coefficient"]: scalar float
            The best friction coefficient determined by the inversion
            or the input friction coefficient (see `friction_coefficient`).
        - output["principal_stresses"]: (3,) numpy.ndarray
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
        - output["principal_directions"]: (3, 3) numpy.ndarray
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i is
            given by: `principal_directions[:, i]`.
        - output["C_m_posterior"]: (5, 5) numpy.ndarray, optional
            Posterior covariance of the model parameter distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["C_d_posterior"]: (3 x n_earthquakes, 3 x n_earthquakes) numpy.ndarray, optional
            Posterior covariance of the data distribution
            estimated from the Tarantola and Valette formula.
            Returned if `return_stats` is True.
        - output["resolution_operator"]: (3, 3) numpy.ndarray
            The resolution operator assuming that the shear tractions are all
            perfectly constant or that they are all perfectly estimated.
    """
    if plot:
        import mplstereonet
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    # compute auxiliary planes
    strikes_1, dips_1, rakes_1 = strikes, dips, rakes
    strikes_2, dips_2, rakes_2 = np.asarray(
        list(map(utils_stress.aux_plane, strikes, dips, rakes))
    ).T
    # define shape variable
    n_earthquakes = strikes_1.shape[0]
    # define flat arrays
    strikes = np.hstack((strikes_1.reshape(-1, 1), strikes_2.reshape(-1, 1))).flatten()
    dips = np.hstack((dips_1.reshape(-1, 1), dips_2.reshape(-1, 1))).flatten()
    rakes = np.hstack((rakes_1.reshape(-1, 1), rakes_2.reshape(-1, 1))).flatten()
    # initialized averaged outputs
    final_stress_tensor = np.zeros((3, 3), dtype=np.float32)
    C_d_posterior = np.zeros((3 * n_earthquakes, 3 * n_earthquakes), dtype=np.float32)
    C_m_posterior = np.zeros((5, 5), dtype=np.float32)
    resolution_operator = np.zeros((5, 5), dtype=np.float32)
    if friction_coefficient is None:
        final_friction_coefficient = 0.0
    for i in range(n_averaging):
        if verbose > 0:
            print(f"-------- {i+1}/{n_averaging} ----------")
        # The stress inversion is sensitive to initial conditions,
        # which are random, especially when dealing with highly
        # noisy focal mechanisms. Therefore, one can repeat the inversion
        # n_averaging times and average the results.
        # -----------------------------------------
        # initialize the average stress tensor array by repeating the
        # stress inversion on n_random_selections datasets drawn by
        # randomly selecting either of the nodal planes as the fault planes
        avg_stress_tensor = np.zeros((3, 3), dtype=np.float32)
        for n in range(n_random_selections):
            nodal_planes = np.random.randint(0, 2, size=n_earthquakes)
            flat_indexes = np.int32(np.arange(n_earthquakes) * 2 + nodal_planes)
            selected_strikes = strikes[flat_indexes]
            selected_dips = dips[flat_indexes]
            selected_rakes = rakes[flat_indexes]
            # invert this subset of nodal planes
            if variable_shear:
                output_ = iterative_linear_si(
                    selected_strikes,
                    selected_dips,
                    selected_rakes,
                    max_n_iterations=200,
                    return_eigen=False,
                    Tarantola_kwargs=Tarantola_kwargs,
                )
            else:
                output_ = Michael1984_inversion(
                    selected_strikes,
                    selected_dips,
                    selected_rakes,
                    return_eigen=False,
                    Tarantola_kwargs=Tarantola_kwargs,
                )
            # add them to the average
            avg_stress_tensor += output_["stress_tensor"]
        avg_stress_tensor /= float(n_random_selections)
        (
            principal_stresses,
            principal_directions,
        ) = utils_stress.stress_tensor_eigendecomposition(avg_stress_tensor)
        R = utils_stress.R_(principal_stresses)
        if verbose > 0:
            print("Initial shape ratio: {:.2f}".format(R))
        if friction_coefficient is None:
            # ---------------------------------------
            #  Repeat the whole inversion for a range of friction coefficients
            #  and keep the inversion results and friction coefficient
            #  that produced the highest average fault plane instability
            # ---------------------------------------
            friction = np.arange(
                friction_min, friction_max + friction_step, friction_step
            )
            n_fric = len(friction)
            Imax = -1000.0
            for j, mu_j in enumerate(friction):
                # run inversion for current value of friction coefficient
                output_j = _stress_inversion_instability(
                    avg_stress_tensor,
                    mu_j,
                    strikes_1,
                    dips_1,
                    rakes_1,
                    strikes_2,
                    dips_2,
                    rakes_2,
                    n_stress_iter=n_stress_iter,
                    Tarantola_kwargs=Tarantola_kwargs,
                    variable_shear=variable_shear,
                    weighted=weighted,
                    max_n_iterations=max_n_iterations,
                    shear_update_atol=shear_update_atol,
                    stress_tensor_update_atol=stress_tensor_update_atol,
                    signed_instability=signed_instability,
                    verbose=verbose,
                    plot=plot,
                )
                # compute eigendecomposition of inverted stress tensor
                (
                    principal_stresses,
                    principal_directions,
                ) = utils_stress.stress_tensor_eigendecomposition(
                    output_j["stress_tensor"]
                )
                R = utils_stress.R_(principal_stresses)
                # compute the average fault plane instability
                I_j = compute_instability_parameter(
                    principal_directions,
                    R,
                    mu_j,
                    strikes_1,
                    dips_1,
                    rakes_1,
                    strikes_2,
                    dips_2,
                    rakes_2,
                    return_fault_planes=False,
                    signed_instability=signed_instability,
                )
                I_j = np.sum(np.max(I_j, axis=-1))
                if I_j > Imax:
                    # new best solution
                    Imax = I_j
                    output_ = output_j.copy()
                    best_friction_coefficient = mu_j
        else:
            # friction coefficient was given by user
            output_ = _stress_inversion_instability(
                avg_stress_tensor,
                friction_coefficient,
                strikes_1,
                dips_1,
                rakes_1,
                strikes_2,
                dips_2,
                rakes_2,
                n_stress_iter=n_stress_iter,
                Tarantola_kwargs=Tarantola_kwargs,
                variable_shear=variable_shear,
                weighted=weighted,
                max_n_iterations=max_n_iterations,
                shear_update_atol=shear_update_atol,
                stress_tensor_update_atol=stress_tensor_update_atol,
                signed_instability=signed_instability,
                verbose=verbose,
                plot=plot,
            )
        final_stress_tensor += output_["stress_tensor"]
        C_d_posterior += output_["C_d_posterior"]
        C_m_posterior += output_["C_m_posterior"]
        resolution_operator += output_["resolution_operator"]
        if friction_coefficient is None:
            # friction coefficient is being inverted for
            final_friction_coefficient += best_friction_coefficient
    final_stress_tensor /= float(n_averaging)
    C_d_posterior /= float(n_averaging)
    C_m_posterior /= float(n_averaging)
    resolution_operator /= float(n_averaging)
    if friction_coefficient is None:
        friction_coefficient = final_friction_coefficient / float(n_averaging)
    # eigendecomposition of averaged stress tensor
    (
        principal_stresses,
        principal_directions,
    ) = utils_stress.stress_tensor_eigendecomposition(final_stress_tensor)
    R = utils_stress.R_(principal_stresses)
    if verbose > 0:
        print("Final results:")
        print("Stress tensor:\n", final_stress_tensor)
        print("Shape ratio: {:.2f}".format(R))
    output = {}
    output["stress_tensor"] = final_stress_tensor
    output["friction_coefficient"] = friction_coefficient
    output["principal_stresses"] = principal_stresses
    output["principal_directions"] = principal_directions
    if return_stats:
        if Tarantola_kwargs is not None:
            output["C_d_posterior"] = C_d_posterior
            output["C_m_posterior"] = C_m_posterior
            output["resolution_operator"] = resolution_operator
        else:
            warnings.warn(
                    "return_stats=True only works with the Tarantola-Valette "
                    "method (see Tarantola_kwargs), returning dummy outputs"
                    )
            output["C_d_posterior"] = np.diag(np.ones(3 * n_earthquakes, dtype=np.float32))
            output["C_m_posterior"] = np.ones((5, 5), dtype=np.float32)
            output["resolution_operator"] = np.ones((3, 3), dtype=np.float32)
    return output


def inversion_jackknife_instability(
    principal_directions,
    R,
    jack_strikes,
    jack_dips,
    jack_rakes,
    friction_coefficient,
    n_resamplings=100,
    n_stress_iter=10,
    stress_tensor_update_atol=1.0e-4,
    Tarantola_kwargs={},
    bootstrap_events=False,
    n_earthquakes=None,
    variable_shear=True,
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    signed_instability=False,
    weighted=False,
    n_threads=1,
    parallel=False,
):
    """
    This routine was tailored for one of my application, but it can
    be of interest to others. Each earthquake comes with an ensemble
    of focal mechanism solutions that were obtained by resampling the
    set of seismic stations used in the focal mechanism inversion. The
    resampling was done with the delete-k-jackknife method, hence the
    name of the routine. This routine randomly samples focal mechanisms
    from these ensembles and runs the stress inversion. This is a way
    of propagating the focal mechanism uncertainties into the stress
    inversion. Use the instability parameter to seek which nodal planes
    are more likely to be the fault planes (cf. B. Lund and R. Slunga 1999,
    V. Vavrycuk 2013,2014).

    Use a previously determined stress tensor (e.g. the output of
    `inversion_one_set_instability`) described by its principal stress
    directions and shape ratio as the prior model in the Tarantola
    and Valette formula. In general, you can keep the default parameter
    values, except for n_resamplings which depends on the time you can
    afford spending.

    Parameters
    -----------
    principal_directions: (3, 3) numpy.ndarray, float
        The three eigenvectors of the reference stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: `principal_directions[:, i]`.
    R: scalar float
        Shape ratio of the reference stress tensor.
    friction_coefficient: scalar float
        Friction value used in the instability parameter. This can be
        the value output by `inversion_one_set_instability`.
    jack_strikes: (n_earthquakes, n_jackknifes) numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    jack_dips: (n_earthquakes, n_jackknifes) numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    jack_rakes: (n_earthquakes, n_jackknifes) numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    n_stress_iter: integer, default to 10
        Maximum number of iterations to seek for the best fault planes.
        See Beauce et al. 2022 for explanations.
    stress_tensor_update_atol: float, default to 1.e-4
        If the RMS difference of the stress tensors between two
        iterations fall below this threshold, convergence has been reached.
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    signed_instability: boolean, default to False
        If True, the instability parameter ranges from -1 to +1. Negative
        values mean that the predicted and observed slip have opposite
        directions. If False, the instability parameter is the one
        defined in Vavrycuk 2013, 2014.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    variable_shear: boolean, default to True
        If True, use the iterative linear method described in
        Beauce et al. 2022, else use the classic linear method
        due to Michael 1984.
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    bootstrap_events: boolean, default to False
        If True, the resampling is also done accross earthquakes,
        following the bootstrapping method.
    weighted: boolean, default to False
        This option is exploratory. If True:
            1) More weight is given to the fault planes that are clearly
               more unstable than their auxiliary counterpart in the
               stress field estimated at iteration t-1
            2) Randomly mixes the set of fault planes at iterations
               t-1 and t, giving larger probability to the planes
               belonging to the set that produced the larger instability.
        This option can be interesting for reaching convergence on
        data sets of bad quality.
    n_threads: scalar int, optional
        Default to `n_threads=1`. If different from 1, the task is parallelized
        across `n_threads` threads. If `n_threads` is `0`, `None` or `"all"`,
        use all available CPUs.

    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["jack_stress_tensor"]: (n_resamplings, 3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["jack_principal_stresses"]: (n_resamplings, 3) numpy.ndarray
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
        - output["jack_principal_directions"]: (n_resamplings, 3, 3) numpy.ndarray
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i for
            the b-th jackknife replica is given by:
            `jack_principal_directions[b, :, i]`.
    """
    # compute auxiliary planes
    jack_strikes_1, jack_dips_1, jack_rakes_1 = jack_strikes, jack_dips, jack_rakes
    jack_strikes_2, jack_dips_2, jack_rakes_2 = np.vectorize(utils_stress.aux_plane)(
        jack_strikes_1, jack_dips_1, jack_rakes_1
    )
    # make a copy of Tarantola_kwargs, on which this function will work
    Tarantola_kwargs_ = Tarantola_kwargs.copy()
    # build reduced stress tensor from principal directions and shape ratio
    stress_tensor_main = utils_stress.reduced_stress_tensor(principal_directions, R)
    sigma_main = np.array(
        [
            stress_tensor_main[0, 0],
            stress_tensor_main[0, 1],
            stress_tensor_main[0, 2],
            stress_tensor_main[1, 1],
            stress_tensor_main[1, 2],
        ]
    ).reshape(-1, 1)
    # define shape variables
    n_earthquakes, n_jackknife = jack_strikes_1.shape
    # initialize the average stress tensor arrays
    jack_stress_tensors = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    jack_principal_stresses = np.zeros((n_resamplings, 3), dtype=np.float32)
    jack_principal_directions = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    jack_strikes_1, jack_dips_1, jack_rakes_1 = (
        jack_strikes_1.flatten(),
        jack_dips_1.flatten(),
        jack_rakes_1.flatten(),
    )
    jack_strikes_2, jack_dips_2, jack_rakes_2 = (
        jack_strikes_2.flatten(),
        jack_dips_2.flatten(),
        jack_rakes_2.flatten(),
    )
    # n_jackknife = None if bootstrap_events else n_jackknife
    n_earthquakes = n_earthquakes if bootstrap_events else None
    _bootstrap_solution_p = partial(
        _bootstrap_solution,
        strikes_1=jack_strikes_1,
        dips_1=jack_dips_1,
        rakes_1=jack_rakes_1,
        strikes_2=jack_strikes_2,
        dips_2=jack_dips_2,
        rakes_2=jack_rakes_2,
        stress_tensor_main=stress_tensor_main,
        friction_coefficient=friction_coefficient,
        stress_tensor_update_atol=stress_tensor_update_atol,
        n_stress_iter=n_stress_iter,
        Tarantola_kwargs=Tarantola_kwargs,
        variable_shear=variable_shear,
        weighted=weighted,
        max_n_iterations=max_n_iterations,
        shear_update_atol=shear_update_atol,
        n_jackknife=n_jackknife,
        n_earthquakes=n_earthquakes,
        signed_instability=signed_instability,
    )
    if parallel:
        print("parallel is deprecated. Use n_threads instead.")
        n_threads = "all"
    if n_threads != 1:
        import concurrent.futures
        if n_threads in [0, None, "all"]:
            # n_threads = None means use all CPUs
            n_threads = None

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(_bootstrap_solution_p, range(n_resamplings)))
        jack_stress_tensors = np.asarray([results[b][0] for b in range(n_resamplings)])
        jack_principal_stresses = np.asarray(
            [results[b][1] for b in range(n_resamplings)]
        )
        jack_principal_directions = np.asarray(
            [results[b][2] for b in range(n_resamplings)]
        )
    else:
        for b in range(n_resamplings):
            if b % 100 == 0:
                print(f"---------- Bootstrapping {b+1}/{n_resamplings} ----------")
            (
                jack_stress_tensors[b, ...],
                jack_principal_stresses[b, ...],
                jack_principal_directions[b, ...],
            ) = _bootstrap_solution_p(b)
    output = {}
    output["jack_stress_tensor"] = jack_stress_tensors
    output["jack_principal_stresses"] = jack_principal_stresses
    output["jack_principal_directions"] = jack_principal_directions
    return output


def inversion_bootstrap_instability(
    principal_directions,
    R,
    strikes,
    dips,
    rakes,
    friction_coefficient,
    n_resamplings=100,
    n_stress_iter=10,
    stress_tensor_update_atol=1.0e-5,
    Tarantola_kwargs={},
    variable_shear=True,
    max_n_iterations=300,
    shear_update_atol=1.0e-5,
    signed_instability=False,
    weighted=False,
    n_threads=1,
    parallel=False,
):
    """
    Invert one set of focal mechanisms with the instability parameter
    to seek which nodal planes are more likely to be the fault planes
    (cf. B. Lund and R. Slunga 1999, V. Vavrycuk 2013,2014).
    Performs bootstrap resampling of the data set to return an
    ensemble of solutions.

    Use a previously determined stress tensor (e.g. the output of
    `inversion_one_set_instability`) described by its principal stress
    directions and shape ratio as the prior model in the Tarantola
    and Valette formula. In general, you can keep the default parameter
    values, except for n_resamplings which depends on the time you can
    afford spending.

    Parameters
    -----------
    principal_directions: (3, 3) numpy.ndarray, float
        The three eigenvectors of the reference stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: `principal_directions[:, i]`.
    R: float
        Shape ratio of the reference stress tensor.
    friction_coefficient: float
        Value of the friction coefficient used in the instability parameter.
        This can be the value output by `inversion_one_set_instability`.
    strikes: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    n_stress_iter: integer, default to 10
        Maximum number of iterations to seek for the best fault planes.
        See Beauce et al. 2022 for explanations.
    stress_tensor_update_atol: float, default to 1.e-4
        If the RMS difference of the stress tensors between two
        iterations fall below this threshold, convergence has been reached.
    shear_update_atol: float, default to 1e-5
        Convergence criterion on the shear stress magnitude updates.
        Convergence is reached when the RMS difference between two
        estimates of shear stress magnitudes falls below that threshold.
    signed_instability: boolean, default to False
        If True, the instability parameter ranges from -1 to +1. Negative
        values mean that the predicted and observed slip have opposite
        directions. If False, the instability parameter is the one
        defined in Vavrycuk 2013, 2014.
    max_n_iterations: integer, default to 300
        The maximum number of iterations if shear stress magnitude update
        does not fall below `shear_update_atol`.
    variable_shear: boolean, default to True
        If True, use the iterative linear method described in
        Beauce et al. 2022, else use the classic linear method
        due to Michael 1984.
    Tarantola_kwargs: Dictionary, default to {}
        If not None, should contain key word arguments
        for the Tarantola and Valette inversion. An empty dictionary
        uses the default values in `Tarantola_Valette`. If None, uses
        the Moore-Penrose inverse.
    weighted: boolean, default to False
        This option is exploratory. If True,
            1) More weight is given to the fault planes that are clearly
               more unstable than their auxiliary counterpart in the
               stress field estimated at iteration t-1
            2) Randomly mixes the set of fault planes at iterations
               t-1 and t, giving larger probability to the planes
               belonging to the set that produced the larger instability.
        This option can be interesting for reaching convergence on
        data sets of bad quality.
    n_threads: scalar int, optional
        Default to `n_threads=1`. If different from 1, the task is parallelized
        across `n_threads` threads. If `n_threads` is `0`, `None` or `"all"`,
        use all available CPUs.

    Returns
    --------
    output: dict {str: numpy.ndarray}
        - output["boot_stress_tensor"]: (n_resamplings, 3, 3) numpy.ndarray
            The inverted stress tensor in the (north, west, up)
            coordinate system.
        - output["boot_principal_stresses"]: (n_resamplings, 3) numpy.ndarray
            The three eigenvalues of the stress tensor, ordered from
            most compressive (sigma1) to least compressive (sigma3).
        - output["boot_principal_directions"]: (n_resamplings, 3, 3) numpy.ndarray
            The three eigenvectors of the stress tensor, stored in a matrix
            as column vectors and ordered from most compressive (sigma1)
            to least compressive (sigma3). The direction of sigma_i for
            the b-th bootstrap replica is given by:
            `boot_principal_directions[b, :, i]`.
    """
    # compute auxiliary planes
    strikes_1, dips_1, rakes_1 = strikes, dips, rakes
    strikes_2, dips_2, rakes_2 = np.asarray(
        list(map(utils_stress.aux_plane, strikes, dips, rakes))
    ).T
    # build reduced stress tensor from principal directions and shape ratio
    stress_tensor_main = utils_stress.reduced_stress_tensor(principal_directions, R)
    sigma_main = np.array(
        [
            stress_tensor_main[0, 0],
            stress_tensor_main[0, 1],
            stress_tensor_main[0, 2],
            stress_tensor_main[1, 1],
            stress_tensor_main[1, 2],
        ]
    ).reshape(-1, 1)
    if Tarantola_kwargs is None:
        Tarantola_kwargs = {}
    Tarantola_kwargs["m_prior"] = sigma_main.copy()
    # define shape variables
    n_earthquakes = len(strikes_1)
    # initialize the average stress tensor arrays
    boot_stress_tensors = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    boot_principal_stresses = np.zeros((n_resamplings, 3), dtype=np.float32)
    boot_principal_directions = np.zeros((n_resamplings, 3, 3), dtype=np.float32)
    _bootstrap_solution_p = partial(
        _bootstrap_solution,
        strikes_1=strikes_1,
        dips_1=dips_1,
        rakes_1=rakes_1,
        strikes_2=strikes_2,
        dips_2=dips_2,
        rakes_2=rakes_2,
        stress_tensor_main=stress_tensor_main,
        friction_coefficient=friction_coefficient,
        stress_tensor_update_atol=stress_tensor_update_atol,
        n_stress_iter=n_stress_iter,
        Tarantola_kwargs=Tarantola_kwargs,
        variable_shear=variable_shear,
        weighted=weighted,
        max_n_iterations=max_n_iterations,
        shear_update_atol=shear_update_atol,
        signed_instability=signed_instability,
    )
    if parallel:
        print("parallel is deprecated. Use n_threads instead.")
        n_threads = "all"
    if n_threads != 1:
        import concurrent.futures
        if n_threads in [0, None, "all"]:
            # n_threads = None means use all CPUs
            n_threads = None

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(_bootstrap_solution_p, range(n_resamplings)))
        boot_stress_tensors = np.asarray([results[b][0] for b in range(n_resamplings)])
        boot_principal_stresses = np.asarray(
            [results[b][1] for b in range(n_resamplings)]
        )
        boot_principal_directions = np.asarray(
            [results[b][2] for b in range(n_resamplings)]
        )
    else:
        for b in range(n_resamplings):
            if b % 100 == 0:
                print(f"---------- Bootstrapping {b+1}/{n_resamplings} ----------")
            (
                boot_stress_tensors[b, ...],
                boot_principal_stresses[b, ...],
                boot_principal_directions[b, ...],
            ) = _bootstrap_solution_p(b)
    output = {}
    output["boot_stress_tensor"] = boot_stress_tensors
    output["boot_principal_stresses"] = boot_principal_stresses
    output["boot_principal_directions"] = boot_principal_directions
    return output


def _bootstrap_solution(
    _,
    strikes_1,
    dips_1,
    rakes_1,
    strikes_2,
    dips_2,
    rakes_2,
    stress_tensor_main,
    friction_coefficient,
    stress_tensor_update_atol,
    n_stress_iter,
    Tarantola_kwargs,
    variable_shear,
    weighted,
    max_n_iterations,
    shear_update_atol,
    n_jackknife=None,
    n_earthquakes=None,
    signed_instability=False,
):
    """Used to parallelize resampling.

    Should not be used directly. If `n_jackknife` is provided, this function
    assumes that strikes, dips, and rakes are given in the "jackknife" format,
    where blocks of `n_jackknifes` contiguous samples are for `n_jackknifes`
    possible solutions of the *same* focal mechanism. Resampling is then
    performed only among the jackknife solutions, and each earthquake still
    appears one time, as in the original data set. Therefore, this resampling
    method only propagates the uncertainties in the focal mechanisms, and not
    the uncertainties related to spatial sampling.
    """
    if n_jackknife is None and n_earthquakes is None:
        bootstrap_set = np.random.choice(
            np.arange(len(strikes_1)), replace=True, size=len(strikes_1)
        )
    elif n_earthquakes is None:
        # default jackknife mode:
        # strikes_1/dips_1/etc... come with contiguous blocks
        # of n_jackknife focal mechanisms that are different
        # possible solutions of the same earthquake, and we
        # only sample from these without sampling with replacement
        # among earthquakes
        n_earthquakes = len(strikes_1) // n_jackknife
        bootstrap_set = np.arange(n_earthquakes) * n_jackknife + np.random.randint(
            0, n_jackknife, size=n_earthquakes
        )
    else:
        # n_earthquakes is specified, bootstrap on jackknife solutions
        # AND on events
        bootstrap_set = np.random.choice(
            np.arange(n_earthquakes), replace=True, size=n_earthquakes
        ) * n_jackknife + np.random.randint(0, n_jackknife, size=n_earthquakes)
    strikes_1_b, dips_1_b, rakes_1_b = (
        strikes_1[bootstrap_set],
        dips_1[bootstrap_set],
        rakes_1[bootstrap_set],
    )
    strikes_2_b, dips_2_b, rakes_2_b = (
        strikes_2[bootstrap_set],
        dips_2[bootstrap_set],
        rakes_2[bootstrap_set],
    )
    output_ = _stress_inversion_instability(
        stress_tensor_main,
        friction_coefficient,
        strikes_1_b,
        dips_1_b,
        rakes_1_b,
        strikes_2_b,
        dips_2_b,
        rakes_2_b,
        n_stress_iter=n_stress_iter,
        Tarantola_kwargs=Tarantola_kwargs,
        variable_shear=variable_shear,
        weighted=weighted,
        max_n_iterations=max_n_iterations,
        shear_update_atol=shear_update_atol,
        stress_tensor_update_atol=stress_tensor_update_atol,
        signed_instability=signed_instability,
        verbose=0,
        plot=False,
    )
    return (output_["stress_tensor"],) + utils_stress.stress_tensor_eigendecomposition(
        output_["stress_tensor"]
    )


def _stress_inversion_instability(
    stress_tensor0,
    friction_coefficient,
    strikes_1,
    dips_1,
    rakes_1,
    strikes_2,
    dips_2,
    rakes_2,
    **kwargs,
):
    """
    Core wrapper function to run the iterative linear stress inversion
    whether shear stress is assumed to be constant (i.e. equivalent to
    the method described in Vavrycuk 2013, 2014) or not (the method described
    in Beauce et al. 2022). This function should be not called directly by
    the user.
    """
    Tarantola_kwargs = kwargs.get("Tarantola_kwargs", {})
    n_stress_iter = kwargs.get("n_stress_iter", 10)
    weighted = kwargs.get("weighted", False)
    variable_shear = kwargs.get("variable_shear", True)
    max_n_iterations = kwargs.get("max_n_iterations", 500)
    shear_update_atol = kwargs.get("shear_update_atol", 1.0e-7)
    signed_instability = kwargs.get("signed_instability", False)
    stress_tensor_update_atol = kwargs.get("stress_tensor_update_atol", 1.0e-4)
    verbose = kwargs.get("verbose", 1)
    plot = kwargs.get("plot", False)
    criterion_on_noconvergence = kwargs.get("criterion_on_noconvergence", "residuals")
    if Tarantola_kwargs is None:
        Tarantola_kwargs = {}
    else:
        # make copy to not overwrite the input dictionary
        Tarantola_kwargs = Tarantola_kwargs.copy()
    # ------------------------------
    #   Start instability criterion
    # ------------------------------
    # initialize variables
    n_earthquakes = len(strikes_1)
    stress_tensor = stress_tensor0
    stress_diff = 0.0
    total_instability = 0.0
    total_differential_instability = -100.0
    instability = 0.1 * np.ones((n_earthquakes, 2), dtype=np.float32)
    residuals = np.finfo(np.float32).max
    best_residuals = np.finfo(np.float32).max
    fault_strikes, fault_dips, fault_rakes = [np.zeros(n_earthquakes) for i in range(3)]
    # C_m_post = np.zeros((5, 5), dtype=np.float32)
    # C_d_post = np.zeros((3 * n_earthquakes, 3 * n_earthquakes), dtype=np.float32)
    weights = np.ones(3 * n_earthquakes, dtype=np.float32)
    # start the nodal plane selection loop
    for n in range(n_stress_iter):
        # get stress state of previous iteration to determine the
        # set of fault planes that maximize instability
        (
            principal_stresses,
            principal_directions,
        ) = utils_stress.stress_tensor_eigendecomposition(stress_tensor0)
        R = utils_stress.R_(principal_stresses)
        # ------------
        # copy variables from previous iteration
        stress_tensor0 = stress_tensor.copy()
        total_instability0 = float(total_instability)
        total_differential_instability0 = float(total_differential_instability)
        instability0 = instability.copy()
        stress_diff0 = float(stress_diff)
        residuals0 = float(residuals)
        fault_strikes0, fault_dips0, fault_rakes0 = (
            fault_strikes.copy(),
            fault_dips.copy(),
            fault_rakes.copy(),
        )
        # -----------
        (
            instability,
            fault_strikes,
            fault_dips,
            fault_rakes,
        ) = compute_instability_parameter(
            principal_directions,
            R,
            friction_coefficient,
            strikes_1,
            dips_1,
            rakes_1,
            strikes_2,
            dips_2,
            rakes_2,
            signed_instability=signed_instability,
            return_fault_planes=True,
        )
        total_instability = np.mean(np.max(instability, axis=-1))
        if weighted:
            # ----------------------------------
            #    This feature is experimental.
            # ----------------------------------
            total_differential_instability = np.mean(
                np.abs(instability[:, 1] - instability[:, 0])
            )
            # decide to keep or not previous fault planes probabilisticly based
            # on the instability values
            # sigmoid probability:
            X = total_differential_instability0 / total_differential_instability - 1.0
            p0 = 1.0 / (1.0 + np.exp(-X))
            # print('Probability: {:.3f} (before: {:.2f}, now: {:.2f})'.format(p0, total_differential_instability0, total_differential_instability))
            R = np.random.random(n_earthquakes)
            fault_strikes = np.float32(
                [
                    fault_strikes0[i] if R[i] < p0 else fault_strikes[i]
                    for i in range(n_earthquakes)
                ]
            )
            fault_dips = np.float32(
                [
                    fault_dips0[i] if R[i] < p0 else fault_dips[i]
                    for i in range(n_earthquakes)
                ]
            )
            fault_rakes = np.float32(
                [
                    fault_rakes0[i] if R[i] < p0 else fault_rakes[i]
                    for i in range(n_earthquakes)
                ]
            )
            instability = np.float32(
                [
                    instability0[i] if R[i] < p0 else instability[i]
                    for i in range(n_earthquakes)
                ]
            )
            # give more weights to focal mechanisms where the most unstable
            # nodal plane is well defined, i.e. has an instability parameter
            # clearly larger than the other plane
            weights = np.repeat(np.abs(instability[:, 1] - instability[:, 0]), 3)
            if weights.sum() == 0.0:
                weights = np.ones(3 * n_earthquakes, dtype=np.float32)
            # normalize the weights such that 1/max(weights) = 0.1 (which is
            # the standard deviation I would give to a good slip measurement)
            weights /= np.median(weights)
            weights = np.clip(weights, 1.0 / np.sqrt(3.0), np.sqrt(3.0))
            weights = 10.0 * weights**2
        else:
            weights = np.ones(3 * n_earthquakes, dtype=np.float32)
        # p = (weights / np.sum(weights))[::3]
        if "C_d" in Tarantola_kwargs:
            # update existing covariance matrix
            Tarantola_kwargs["C_d"] = Tarantola_kwargs["C_d"] + np.diag(1.0 / weights)
        elif "C_d" in Tarantola_kwargs and weighted:
            # keep previous weights in memory
            Tarantola_kwargs["C_d"] = 0.7 * Tarantola_kwargs["C_d"] + 0.3 * np.diag(
                1.0 / weights
            )
        else:
            Tarantola_kwargs["C_d"] = np.diag(1.0 / weights)
        Tarantola_kwargs["C_d_inv"] = np.linalg.inv(Tarantola_kwargs["C_d"])
        if variable_shear:
            output_ = iterative_linear_si(
                fault_strikes,
                fault_dips,
                fault_rakes,
                return_eigen=False,
                return_stats=True,
                Tarantola_kwargs=Tarantola_kwargs,
                max_n_iterations=max_n_iterations,
                shear_update_atol=shear_update_atol,
            )
        else:
            output_ = Michael1984_inversion(
                fault_strikes,
                fault_dips,
                fault_rakes,
                return_eigen=False,
                return_stats=True,
                Tarantola_kwargs=Tarantola_kwargs,
            )
        stress_tensor = output_["stress_tensor"]
        stress_diff = np.sum((output_["stress_tensor"] - stress_tensor0) ** 2)
        if criterion_on_noconvergence == "residuals":
            # ------------------------------------
            # Compute residuals in case the instability loop doesn't converge
            # ------------------------------------
            # normal and slip vectors
            n_, d_ = utils_stress.normal_slip_vectors(
                fault_strikes, fault_dips, fault_rakes
            )
            shear_mag = np.sqrt(np.sum(output_["predicted_shear_stress"] ** 2, axis=-1))
            if variable_shear:
                res = (
                    output_["predicted_shear_stress"] - shear_mag[:, np.newaxis] * d_.T
                ).reshape(-1, 1)
            else:
                res = (
                    output_["predicted_shear_stress"] - np.mean(shear_mag) * d_.T
                ).reshape(-1, 1)
            residuals = (res.T @ Tarantola_kwargs["C_d_inv"] @ res)[0, 0]
            if residuals < best_residuals:
                # One possibility: update prior model at this stage
                # Tarantola_kwargs["m_prior"] = np.array(
                #    [
                #        stress_tensor[0, 0],
                #        stress_tensor[0, 1],
                #        stress_tensor[0, 2],
                #        stress_tensor[1, 1],
                #        stress_tensor[1, 2],
                #    ]
                # ).reshape(-1, 1)
                # store best results
                best_residuals = float(residuals)
                best_stress_tensor = output_["stress_tensor"].copy()
                best_C_m_post = output_["C_m_posterior"].copy()
                best_C_d_post = output_["C_d_posterior"].copy()
                best_resolution_operator = output_["resolution_operator"].copy()
        if plot:
            fig = plt.figure("iteration_{:d}".format(n))
            ax1 = fig.add_subplot(2, 2, 1, projection="stereonet")
            ax1.set_title("R={:.2f}".format(R))
            markers = ["o", "s", "v"]
            for k in range(3):
                az, pl = utils_stress.get_bearing_plunge(
                    output_["principal_directions[:, k]"]
                )
                ax1.line(
                    pl,
                    az,
                    marker=markers[k],
                    markeredgecolor="k",
                    color=f"C{k}",
                    markersize=15,
                )
            ax2 = fig.add_subplot(2, 2, 2, projection="stereonet")
            cNorm = Normalize(vmin=0.0, vmax=p.max())
            scalar_map = ScalarMappable(norm=cNorm, cmap="cividis")
            # ax2.plane(fault_strikes, fault_dips, color=scalar_map.to_rgba(p), lw=2.0)
            for ii in range(len(fault_strikes)):
                ax2.plane(
                    fault_strikes[ii],
                    fault_dips[ii],
                    color=scalar_map.to_rgba(p[ii]),
                    lw=2.0,
                )
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.08, axes_class=plt.Axes)
            plt.colorbar(scalar_map, cax=cax, label="Weight", orientation="vertical")
            plt.show(block=True)
        if verbose == 2:
            print("----------")
            print(
                "Stress tensor difference at iteration {:d}: {}.".format(n, stress_diff)
            )
            # print(stress_tensor)
            (
                principal_stresses,
                principal_directions,
            ) = utils_stress.stress_tensor_eigendecomposition(output_["stress_tensor"])
            R = utils_stress.R_(principal_stresses)
            print("R={:.2f}, friction={:.2f}".format(R, friction_coefficient))
            print(
                "Total instability: {:.2f}/Total differential instability: {:.2f}".format(
                    total_instability, total_differential_instability
                )
            )
            print(
                "Average angle: {:.2f}".format(
                    utils_stress.mean_angular_residual(
                        output_["stress_tensor"], fault_strikes, fault_dips, fault_rakes
                    )
                )
            )
            print("Squared residuals: {:.2e}".format(residuals))
        if stress_diff < stress_tensor_update_atol:
            # stop stress instability loop
            break
    output = {}
    if stress_diff >= stress_tensor_update_atol:
        # did not convergence, get results from best stress tensor
        output["stress_tensor"] = best_stress_tensor
        output["C_m_posterior"] = best_C_m_post
        output["C_d_posterior"] = best_C_d_post
        output["resolution_operator"] = best_resolution_operator
        if verbose > 0:
            (
                principal_stresses,
                principal_directions,
            ) = utils_stress.stress_tensor_eigendecomposition(output["stress_tensor"])
            R = utils_stress.R_(principal_stresses)
            print("Did not converge, return best (R={:.2f})".format(R))
    else:
        # output is the result of the last iteration
        output = output_
    return output


# ---------------------------------------------------
#
#          Routines for instability criterion
#
# ---------------------------------------------------


def find_optimal_friction(
    strikes_1,
    dips_1,
    rakes_1,
    strikes_2,
    dips_2,
    rakes_2,
    principal_directions,
    R,
    friction_min=0.1,
    friction_max=0.8,
    friction_step=0.05,
    signed_instability=False,
):
    """
    Find the friction that maximizes the instability parameter I
    based on V. Vavrycuk 2013,2014 and B. Lund and R. Slunga 1999.

    Parameters
    -----------
    strikes_1: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips_1: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes_1: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    strikes_2: list or numpy.ndarray, float
        The strike of nodal planes 2, angle between north and
        the fault's horizontal (0-360).
    dips_2: list or numpy.ndarray, float
        The dip of nodal planes 2, angle between the horizontal
        plane and the fault plane (0-90).
    rakes_2: list or numpy.ndarray, float
        The rake of nodal planes 2, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    principal_directions: (3, 3) numpy.ndarray, float
        The three eigenvectors of the reference stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: `principal_directions[:, i]`.
    R: float
        Shape ratio of the reference stress tensor.
    friction_min: float, default to 0.1
        Lower bound of explored friction values.
    friction_max: float, default to 0.8
        Upper bound of explored friction values.
    friction_step: float, default to 0.05
        Step employed in the grid search of the friction value
        that maximizes the instability parameter.
    signed_instability: boolean, default to False
        If True, the instability parameter ranges from -1 to +1. Negative
        values mean that the predicted and observed slip have opposite
        directions. If False, the instability parameter is the one
        defined in Vavrycuk 2013, 2014.

    Returns
    --------
    optimal_friction: float
        The friction value that maximizes the mean instability parameter.
    """
    friction = np.arange(friction_min, friction_max + friction_step, friction_step)
    n_fric = len(friction)
    I = np.zeros(n_fric, dtype=np.float32)
    for i, fric in enumerate(friction):
        I_ = compute_instability_parameter(
            principal_directions,
            R,
            fric,
            strikes_1,
            dips_1,
            rakes_1,
            strikes_2,
            dips_2,
            rakes_2,
            signed_instability=signed_instability,
        )
        I[i] = np.sum(np.max(I_, axis=-1))
    optimal_friction = friction[I.argmax()]
    return optimal_friction


def find_optimal_friction_one_set(
    strikes_1,
    dips_1,
    rakes_1,
    principal_directions,
    R,
    friction_min=0.2,
    friction_max=0.8,
    friction_step=0.05,
    signed_instability=False,
):
    """
    Find the friction that maximizes the instability parameter I
    based on V. Vavrycuk 2013,2014 and B. Lund and R. Slunga 1999.

    Parameters
    -----------
    strikes_1: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips_1: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes_1: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    principal_directions: (3, 3) numpy.ndarray, float
        The three eigenvectors of the reference stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: `principal_directions[:, i]`.
    R: float
        Shape ratio of the reference stress tensor.
    friction_min: float, default to 0.1
        Lower bound of explored friction values.
    friction_max: float, default to 0.8
        Upper bound of explored friction values.
    friction_step: float, default to 0.05
        Step employed in the grid search of the friction value
        that maximizes the instability parameter.
    signed_instability: boolean, default to False
        If True, the instability parameter ranges from -1 to +1. Negative
        values mean that the predicted and observed slip have opposite
        directions. If False, the instability parameter is the one
        defined in Vavrycuk 2013, 2014.

    Returns
    --------
    optimal_friction: float
        The friction value that maximizes the mean instability parameter.
    """

    # fake nodal planes #2
    strikes_2, dips_2, rakes_2 = [np.zeros(len(strikes_1)) for i in range(3)]
    friction = np.arange(friction_min, friction_max + friction_step, friction_step)
    n_fric = len(friction)
    I = np.zeros(n_fric, dtype=np.float32)
    for i, fric in enumerate(friction):
        I_ = compute_instability_parameter(
            principal_directions,
            R,
            fric,
            strikes_1,
            dips_1,
            rakes_1,
            strikes_2,
            dips_2,
            rakes_2,
            signed_instability=signed_instability,
        )
        # only look at instability on nodal planes #1
        I[i] = np.sum(I_[:, 0])
    optimal_friction = friction[I.argmax()]
    return optimal_friction


def compute_instability_parameter(
    principal_directions,
    R,
    friction,
    strike_1,
    dip_1,
    rake_1,
    strike_2,
    dip_2,
    rake_2,
    return_fault_planes=False,
    signed_instability=False,
):
    """
    Compute the instability parameter as introduced by Lund and Slunga 1999,
    re-used by Vavrycuk 2013-2014 and modified by Beauce 2022.
    For a given stress field characterized by the principal stress
    directions and shape ratio R=(sig1-sig2)/(sig1-sig3), and for
    a given rock friction, this routine computes an instability
    parameter based on the Mohr-Coulomb failure criterion to determine
    which of the two nodal planes of a focal mechanism solution
    is more likely to be the fault plane.
    Beauce 2022 includes the sign of the dot product between
    the shear stress and the slip vector on the fault. This instability
    ranges from -1 to +1, instead of from 0 to +1.

    Parameters
    -----------
    principal_directions: (3, 3) numpy.ndarray, float
        The three eigenvectors of the reference stress tensor, stored in
        a matrix as column vectors and ordered from
        most compressive (sigma1) to least compressive (sigma3).
        The direction of sigma_i is given by: principal_directions[:, i]
    R: float
        Shape ratio of the reference stress tensor.
    strikes_1: list or numpy.ndarray, float
        The strike of nodal planes 1, angle between north and
        the fault's horizontal (0-360).
    dips_1: list or numpy.ndarray, float
        The dip of nodal planes 1, angle between the horizontal
        plane and the fault plane (0-90).
    rakes_1: list or numpy.ndarray, float
        The rake of nodal planes 1, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    strikes_2: list or numpy.ndarray, float
        The strike of nodal planes 2, angle between north and
        the fault's horizontal (0-360).
    dips_2: list or numpy.ndarray, float
        The dip of nodal planes 2, angle between the horizontal
        plane and the fault plane (0-90).
    rakes_2: list or numpy.ndarray, float
        The rake of nodal planes 2, angle between the fault's horizontal
        and the slip direction of the hanging wall w.r.t. the
        foot wall (0-360 or -180-180).
    return_fault_planes: boolean, default to False
        If True, return the strikes, dips, rakes of the selected
        fault planes.
    signed_instability: boolean, default to False
        If True, the instability parameter ranges from -1 to +1. Negative
        values mean that the predicted and observed slip have opposite
        directions. If False, the instability parameter is the one
        defined in Vavrycuk 2013, 2014.

    Returns
    --------
    instability_parameter: (n_earthquakes, 2) numpy.ndarray
        The instability parameter as defined in Beauce 2022 for the two
        nodal planes of each focal mechanism datum.
    strikes: list or numpy.ndarray, float, optional
        Strikes of the fault planes with largest instability.
        Only provided if `return_fault_planes=True`.
    dips: list or numpy.ndarray, float, optional
        Dips of the fault planes with largest instability.
        Only provided if `return_fault_planes=True`.
    rakes: list or numpy.ndarray, float, optional
        Rakes of the fault planes with largest instability.
        Only provided if `return_fault_planes=True`.
    """
    # the calculation is done in the eigenbasis, therefore
    # we need to project the normal vectors onto the eigenbasis
    n_earthquakes = len(strike_1)
    n_1 = np.zeros((n_earthquakes, 3), dtype=np.float32)
    n_2 = np.zeros((n_earthquakes, 3), dtype=np.float32)
    d_1 = np.zeros((n_earthquakes, 3), dtype=np.float32)
    d_2 = np.zeros((n_earthquakes, 3), dtype=np.float32)
    # compute the normals of the two nodal planes
    for i in range(n_earthquakes):
        n_1[i, :], d_1[i, :] = utils_stress.normal_slip_vectors(
            strike_1[i], dip_1[i], rake_1[i], direction="inward"
        )
        n_2[i, :], d_2[i, :] = utils_stress.normal_slip_vectors(
            strike_2[i], dip_2[i], rake_2[i], direction="inward"
        )
    # project the normals onto the eigenbasis
    n_1 = np.dot(n_1, principal_directions)
    n_2 = np.dot(n_2, principal_directions)
    d_1 = np.dot(d_1, principal_directions)
    d_2 = np.dot(d_2, principal_directions)
    # when parameterazing the reduced stress tensor as:
    # sigma_1 = -1, sigma_2 = 2R-1 and sigma_3 = +1,
    # (CONVENTION: TENSION IS POSITIVE)
    # the critical shear and normal stresses are:
    sig1 = -1.0
    sig2 = 2.0 * R - 1.0
    sig3 = +1.0
    tau_c = 1.0 / np.sqrt(1.0 + friction**2)
    sig_c = friction / np.sqrt(1.0 + friction**2)
    # print(sig1, sig2, sig3, tau_c, sig_c)
    # and the shear and normal stresses on each fault are:
    normal_1 = sig1 * n_1[:, 0] ** 2 + sig2 * n_1[:, 1] ** 2 + sig3 * n_1[:, 2] ** 2
    shear_1 = np.sqrt(
        sig1**2 * n_1[:, 0] ** 2
        + sig2**2 * n_1[:, 1] ** 2
        + sig3**2 * n_1[:, 2] ** 2
        - normal_1**2
    )

    normal_2 = sig1 * n_2[:, 0] ** 2 + sig2 * n_2[:, 1] ** 2 + sig3 * n_2[:, 2] ** 2
    shear_2 = np.sqrt(
        sig1**2 * n_2[:, 0] ** 2
        + sig2**2 * n_2[:, 1] ** 2
        + sig3**2 * n_2[:, 2] ** 2
        - normal_2**2
    )
    # combine all of them in the definition of the instability parameter I
    Ic = tau_c - friction * (sig1 - sig_c)
    I_1 = (shear_1 - friction * (sig1 - normal_1)) / Ic
    I_2 = (shear_2 - friction * (sig1 - normal_2)) / Ic
    # my addition: add the sign of the shear-slip dot product
    # stress_tensor = np.dot(principal_directions, np.dot(np.diag([sig1, sig2, sig3]), principal_directions.T))
    stress_tensor = np.diag([sig1, sig2, sig3])
    traction_1_vec = np.dot(stress_tensor, n_1.T).T
    traction_2_vec = np.dot(stress_tensor, n_2.T).T
    normal_1_vec = np.sum(traction_1_vec * n_1, axis=-1)[:, np.newaxis] * n_1
    normal_2_vec = np.sum(traction_2_vec * n_2, axis=-1)[:, np.newaxis] * n_2
    shear_1_vec = traction_1_vec - normal_1_vec
    shear_2_vec = traction_2_vec - normal_2_vec
    sign_dot_1 = np.sign(np.sum(shear_1_vec * d_1, axis=-1))
    sign_dot_2 = np.sign(np.sum(shear_2_vec * d_2, axis=-1))
    # print(sign_dot_1)
    if signed_instability:
        # multiplying the instability parameter by the sign of the
        # dot product between shear direction and slip direction is
        # the difference with the instability parameter defined in Vavrycuk 2013
        I_1 *= sign_dot_1
        I_2 *= sign_dot_2

    if return_fault_planes:
        strikes = np.zeros(n_earthquakes, dtype=np.float32)
        dips = np.zeros(n_earthquakes, dtype=np.float32)
        rakes = np.zeros(n_earthquakes, dtype=np.float32)
        mask_1 = I_1 >= I_2
        strikes[mask_1] = strike_1[mask_1]
        dips[mask_1] = dip_1[mask_1]
        rakes[mask_1] = rake_1[mask_1]
        mask_2 = ~mask_1
        strikes[mask_2] = strike_2[mask_2]
        dips[mask_2] = dip_2[mask_2]
        rakes[mask_2] = rake_2[mask_2]
        return np.column_stack((I_1, I_2)), strikes, dips, rakes
    else:
        return np.column_stack((I_1, I_2))
