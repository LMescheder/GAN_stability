import numpy as np
from matplotlib import pyplot as plt


def arrow_plot(x, y, color='C1'):
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1],
               color=color, scale_units='xy', angles='xy', scale=1)


def vector_field_plot(theta, psi, v1, v2, trajectory=None, clip_y=None, marker='b^'):
    plt.quiver(theta, psi, v1, v2)
    if clip_y is not None:
        plt.axhspan(np.min(psi), -clip_y, facecolor='0.2', alpha=0.5)
        plt.plot([np.min(theta), np.max(theta)], [-clip_y, -clip_y], 'k-')
        plt.axhspan(clip_y, np.max(psi), facecolor='0.2', alpha=0.5)
        plt.plot([np.min(theta), np.max(theta)], [clip_y, clip_y], 'k-')

    if trajectory is not None:
        psis, thetas = trajectory
        plt.plot(psis, thetas, marker, markerfacecolor='None')
        plt.plot(psis[0], thetas[0], 'ro')

    plt.xlim(np.min(theta), np.max(theta))
    plt.ylim(np.min(psi), np.max(psi))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\psi$')
    plt.xticks(np.linspace(np.min(theta), np.max(theta), 5))
    plt.yticks(np.linspace(np.min(psi), np.max(psi), 5))
