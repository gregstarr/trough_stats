import numpy as np
from matplotlib import pyplot as plt


def format_polar_mag_ax(ax, tick_color='black', with_ticklabels=True):
    if isinstance(ax, np.ndarray):
        for a in ax.flatten():
            format_polar_mag_ax(a)
    else:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                       labeltop=False, labelleft=False, labelright=False)
        ax.set_ylim(0, 60)
        if with_ticklabels:
            ax.set_xticks(np.arange(8) * np.pi / 4)
            ax.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
            ax.set_yticks([10, 20, 30, 40, 50, 60])
            ax.set_yticklabels([80, 70, 60, 50, 40, 30])
        else:
            ax.set_xticks(np.arange(8) * np.pi / 4)
            ax.set_xticklabels([])
            ax.set_yticks([10, 20, 30, 40, 50, 60])
            ax.set_yticklabels([])
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, colors=tick_color)
        ax.tick_params(axis='y', which='both', left=True, labelleft=True, width=0, length=0, colors=tick_color)
        ax.set_rlabel_position(70)
        ax.grid(True, alpha=.5)


def plot_polar_scatter(mlat, mlt, data, size=5, marker='.', figsize=(6, 6), publication=False, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
    ax.set_facecolor('grey')
    theta = np.pi * (mlt - 6) / 12
    rad = 90 - mlat
    ax.scatter(theta, rad, size, data, marker, **kwargs)
    format_polar_mag_ax(ax, with_ticklabels=(not publication))
    return fig, ax


def plot_polar_lines(data, cmap=None):
    fig, ax = plt.subplots(subplot_kw={'polar': True})

    mlt_key = [dim for dim in data.dims if 'mlt' in dim][0]
    theta = np.pi * (data.coords[mlt_key] - 6) / 12
    rad = (90 - data).assign_coords(theta=theta)

    if data.ndim > 1:
        hue_key = [dim for dim in data.dims if 'mlt' not in dim][0]
        lines = rad.plot.line(ax=ax, x='theta', hue=hue_key, add_legend=False)
        if cmap is not None:
            if data[hue_key].dtype == object:
                hue_vals = np.array([h.item().mid for h in data[hue_key]])
            else:
                hue_vals = data[hue_key].values
            hue_vals -= np.min(hue_vals)
            hue_vals /= np.max(hue_vals)
            colors = plt.get_cmap(cmap)(hue_vals)
            for i, line in enumerate(lines):
                line.set_color(colors[i])
    else:
        rad.plot.line(ax=ax, x='theta', add_legend=False)
    format_polar_mag_ax(ax)
    return fig, ax


def plot_polar_pcolormesh(data, figsize=(8, 6), publication=False, **kwargs):
    if data.ndim == 3:
        extra_dim = [dim for dim in data.dims if 'mlt' not in dim and 'mlat' not in dim][0]
        for i in range(data[extra_dim].shape[0]):
            plot_polar_pcolormesh(data.isel({extra_dim: i}))
    elif data.ndim == 2:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
        ax.set_facecolor('grey')
        theta = np.pi * (data.mlt - 6) / 12
        rad = 90 - data.mlat
        pcm = ax.pcolormesh(theta, rad, data.values, shading='auto', **kwargs)
        plt.colorbar(pcm)
        format_polar_mag_ax(ax, with_ticklabels=(not publication))
        return fig, ax
    else:
        raise Exception
