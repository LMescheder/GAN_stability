import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
from diracgan.gans import WGAN
from diracgan.subplots import vector_field_plot
from tqdm import tqdm


def plot_vector(vecfn, theta, psi, outfile, trajectory=None, marker='b^'):
    fig, ax = plt.subplots(1, 1)
    theta, psi = np.meshgrid(theta, psi)
    v1, v2 = vecfn(theta, psi)
    if isinstance(vecfn, WGAN):
        clip_y = vecfn.clip
    else:
        clip_y = None
    vector_field_plot(theta, psi, v1, v2, trajectory, clip_y=clip_y, marker=marker)
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()


def simulate_trajectories(vecfn, theta, psi, trajectory, outfolder, maxframes=300):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    theta, psi = np.meshgrid(theta, psi)
    
    N = min(len(trajectory[0]), maxframes)

    v1, v2 = vecfn(theta, psi)
    if isinstance(vecfn, WGAN):
        clip_y = vecfn.clip
    else:
        clip_y = None

    for i in tqdm(range(1, N)):
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                       subplot_kw=dict(adjustable='box', aspect=0.7))

        plt.sca(ax1)
        trajectory_i = [trajectory[0][:i], trajectory[1][:i]]
        vector_field_plot(theta, psi, v1, v2, trajectory_i, clip_y=clip_y, marker='b-')
        plt.plot(trajectory_i[0][-1], trajectory_i[1][-1], 'bo')

        plt.sca(ax2)
        ax2.set_axisbelow(True)
        plt.grid()

        x = np.linspace(np.min(theta), np.max(theta))
        y = x*trajectory[1][i]
        plt.plot(x, y, 'C1')

        ax2.add_patch(patches.Rectangle(
                (-0.05, 0), .1, 2.5, facecolor='C2'
        ))

        ax2.add_patch(patches.Rectangle(
                (trajectory[0][i]-0.05, 0), .1, 2.5, facecolor='C0'
        ))

        plt.xlim(np.min(theta), np.max(theta))
        plt.ylim(-1, 3.)
        plt.xlabel(r'$\theta$')
        plt.xticks(np.linspace(np.min(theta), np.max(theta), 5))
        ax2.set_yticklabels([])

        plt.savefig(os.path.join(outfolder, '%06d.png' % i), dpi=200, bbox_inches='tight')
        plt.close()
