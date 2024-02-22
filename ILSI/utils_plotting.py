import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_resolution_covariance_corrcoef(
    R,
    C_m_post,
    C_m_post_normalized,
    figname="resolution_covariance",
    figtitle=r"Resolution operator $R = LG$, posterior covariance $\hat{C}_M$"
             r" and normalized posterior covariance $\rho_M$",
    figsize=(13, 10),
    cmap="viridis"
):
    """Plot resolution operator, posterior covariance, and normalized posterior covariance.

    Parameters:
    -----------
    R : array_like
        Resolution operator.
    C_m_post : array_like
        Posterior covariance.
    C_m_post_normalized : array_like
        Normalized posterior covariance.
    figname : str, optional
        Figure name (default is 'resolution_covariance').
    figtitle : str, optional
        Figure title (default is 'Resolution operator R = LG, posterior
        covariance C_M, and normalized posterior covariance ρ_M').
    figsize : tuple, optional
        Figure size (default is (13, 10)).
    cmap : str or Colormap, optional
        Colormap for plotting (default is 'viridis').
    """
    fig, axes = plt.subplots(num=figname, ncols=2, nrows=2, figsize=figsize)

    fig.suptitle(figtitle)
    plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.30)

    axes[0, 0].set_title(r"Resolution operator, $R = LG$")
    pc0 = axes[0, 0].pcolormesh(R, cmap=cmap, rasterized=True,)
    plt.colorbar(pc0, label="Resolution operator")

    diff = np.abs(np.identity(5) - R)
    axes[0, 1].set_title(r"$\vert I - R \vert$")
    pc1 = axes[0, 1].pcolormesh(diff, cmap=cmap, rasterized=True, vmin=0.,)
    plt.colorbar(pc1, label="Deviation from identity")

    axes[1, 0].set_title(r"Posterior covariance operator, $\hat{C}_M$")
    pc2 = axes[1, 0].pcolormesh(C_m_post, cmap=cmap, rasterized=True)
    plt.colorbar(pc2, label="Covariance")

    vmin = min(C_m_post_normalized[C_m_post_normalized < 0.999].min(), -C_m_post_normalized[C_m_post_normalized < 0.999].max())+0.01
    vmax = -vmin
    # vmin, vmax = -1., +1.
    cm = plt.cm.get_cmap("coolwarm")
    cm.set_over("w")

    axes[1, 1].set_title(r"Normalized post. covariance op., $\rho_M$")
    pc3 = axes[1, 1].pcolormesh(
            C_m_post_normalized, rasterized=True, vmin=vmin, vmax=vmax, cmap=cm
            )
    plt.colorbar(pc3, label="Normalized covariance")

    tickpos = [0.5, 1.5, 2.5, 3.5, 4.5]
    ticklabels = [
            r"$\sigma_{11}$",
            r"$\sigma_{12}$",
            r"$\sigma_{13}$",
            r"$\sigma_{22}$",
            r"$\sigma_{23}$"
            ]
    for i, ax in enumerate(axes.flatten()):
        ax.set_xlabel("True model parameter")
        ax.set_ylabel("Inverted model parameter")
        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(tickpos)
        ax.set_yticklabels(ticklabels)
        ax.text(
            -0.1,
            1.05,
            f"({string.ascii_lowercase[i]})",
            transform=ax.transAxes,
            size=20,
        )
    return fig

