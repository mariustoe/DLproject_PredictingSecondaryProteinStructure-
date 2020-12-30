import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.display import Image, display, clear_output
from sklearn.manifold import TSNE
from torch import Tensor
from torch.distributions import Normal
from torchvision.utils import make_grid


def plot_2d_latents(ax, qz, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    scale_factor = 2
    batch_size = z.shape[0]
    palette = sns.color_palette()
    colors = [palette[l] for l in y]

    # plot prior
    prior = plt.Circle((0, 0), scale_factor, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior)

    # plot data points
    mus, sigmas = qz.mu.to('cpu'), qz.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(z[:, 0], z[:, 1], color=colors)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal', 'box')


def plot_latents(ax, z, y):
    z = z.to('cpu')
    palette = sns.color_palette()
    colors = [palette[l] for l in y]
    z = TSNE(n_components=2).fit_transform(z)
    ax.scatter(z[:, 0], z[:, 1], color=colors)


def make_vae_plots(vae, x, y, outputs, training_data, validation_data, figsize=(18, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)

    # plot the observation
    #axes[0, 0].set_title(r'Observation $\mathbf{x}$')
    #plot_samples(axes[0, 0], x)



    # plot posterior samples
    #axes[0, 2].set_title(
    #    r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
    #px = outputs['px']
    #x_sample = px.sample().to('cpu')
    #plot_samples(axes[0, 2], x_sample)

    # plot ELBO
    ax = axes[0, 0]
    ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
    ax.plot(training_data['elbo'], label='Training')
    ax.plot(validation_data['elbo'], label='Validation')
    ax.legend()

    # plot KL
    ax = axes[0, 1]
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
    ax.plot(training_data['kl'], label='Training')
    ax.plot(validation_data['kl'], label='Validation')
    ax.legend()

    # plot NLL
    ax = axes[0, 2]
    ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
    ax.plot(training_data['log_px'], label='Training')
    ax.plot(validation_data['log_px'], label='Validation')
    ax.legend()

    # plot prior samples
    #axes[2, 0].set_title(r'Samples $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim p(\mathbf{z})$')
    #px = vae.sample_from_prior(batch_size=x.size(0))['px']
    #x_samples = px.sample()
    #plot_samples(axes[2, 0], x_samples)

    # plot interpolations samples
    #axes[2, 1].set_title(
    #    r'Latent Interpolations: $\mathbf{x} \sim p_\theta(\mathbf{x} | t \cdot \mathbf{z}_1 + (1-t) \cdot \mathbf{z}_2), \mathbf{z}_1, \mathbf{z}_2 \sim p(\mathbf{z}), t=0 \dots 1$')
    #plot_interpolations(axes[2, 1], vae)

    # plot samples (sampling from a grid instead of the prior)
    #if vae.latent_features == 2:
    #    axes[2, 2].set_title(
    #        r'Samples: $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim \operatorname{grid}(-3:3, -3:3)$')
    #    px = vae.sample_from_prior(batch_size=x.size(0))['px']
    #    x_samples = px.sample()
    #    plot_grid(axes[2, 2], vae)

    # display
    plt.tight_layout()
    #plt.close(fig)
    #clear_output(wait=True)

    plt.savefig("loss_imgs.png")


    #
