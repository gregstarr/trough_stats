import pandas
import pathlib
import numpy as np
from scipy.stats import binned_statistic, linregress
from matplotlib import pyplot as plt
import h5py

from ttools import io, utils, config

from get_dataset import get_tec_dataset

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_swarm_troughs(threshold):
    swarm_trough_dir = pathlib.Path("E:\\ttools\\swarm\\trough candidates")

    swarm_troughs = []
    for p in swarm_trough_dir.glob("*.h5"):
        print(p)
        all_troughs = pandas.read_hdf(p, 'troughs')
        swarm_troughs.append(all_troughs)
    swarm_troughs = pandas.concat(swarm_troughs, ignore_index=True)
    swarm_troughs.direction = swarm_troughs.direction == 'up'
    sat = np.zeros(swarm_troughs.shape[0], dtype=int)
    for i, s in enumerate(['A', 'B', 'C']):
        m = swarm_troughs.sat == s
        sat[m] = i
    swarm_troughs.sat = sat
    swarm_troughs.tec_time = utils.datetime64_to_timestamp(swarm_troughs.tec_time)
    d = swarm_troughs.tec_time.values.astype('datetime64[s]').astype('datetime64[D]')
    t = (d - d[0]) / np.timedelta64(1, 'D')
    swarm_troughs['sat_ind'] = (144 * t + 6 * swarm_troughs.tec_ind + 2 * swarm_troughs.sat + swarm_troughs.direction).values
    yes_troughs = swarm_troughs[swarm_troughs.min_dne <= threshold]
    all_unique_ids, all_unique_counts = np.unique(swarm_troughs.sat_ind, return_counts=True)
    yes_unique_ids, yes_unique_idx = np.unique(yes_troughs.sat_ind, axis=0, return_index=True)
    yes_troughs = yes_troughs.iloc[yes_unique_idx]
    no_troughs = swarm_troughs[np.isin(swarm_troughs.sat_ind, all_unique_ids[all_unique_counts == 1])]
    return pandas.concat([no_troughs, yes_troughs]).sort_index()


def plot_swarm(ax):
    swarm_troughs = get_swarm_troughs(-.2)
    swarm_troughs = swarm_troughs[(swarm_troughs.min_mlt > -5) & (swarm_troughs.min_mlt < 5)]
    swarm_kp = io.get_kp(swarm_troughs.tec_time.values)
    yes_mask = swarm_troughs.trough
    width = abs(swarm_troughs.e1_mlat - swarm_troughs.e2_mlat)[yes_mask]
    min_mlat = swarm_troughs.min_mlat[yes_mask]

    mean_stat = binned_statistic(swarm_kp, yes_mask, 'mean', np.arange(10))
    ax[0].bar(np.arange(9) + .3, mean_stat.statistic, .4, color=colors[0])

    mean_stat = binned_statistic(swarm_kp[yes_mask], min_mlat, 'mean', np.arange(10))
    std_stat = binned_statistic(swarm_kp[yes_mask], min_mlat, 'std', np.arange(10))
    reg = linregress(swarm_kp[yes_mask], min_mlat)
    x = np.array([0, 9])
    y = reg.slope * x + reg.intercept
    ax[1].plot(x, y, '-', c=colors[0], label='Aa 2020')
    ax[1].errorbar(np.arange(9) + .4, mean_stat.statistic, yerr=std_stat.statistic, fmt='o', c=colors[0])

    mean_stat = binned_statistic(swarm_kp[yes_mask], width, 'mean', np.arange(10))
    std_stat = binned_statistic(swarm_kp[yes_mask], width, 'std', np.arange(10))
    ax[2].errorbar(np.arange(9) + .4, mean_stat.statistic, yerr=std_stat.statistic, fmt='o-', c=colors[0])

    # mean_stat = binned_statistic(swarm_kp[yes_mask], depth, 'mean', np.arange(10))
    # std_stat = binned_statistic(swarm_kp[yes_mask], depth, 'std', np.arange(10))
    # ax[3].errorbar(np.arange(9) + .6, mean_stat.statistic, yerr=std_stat.statistic, fmt='o-', c=colors[0])


def plot_tec(ax, score_dir, threshold):
    times, tec, troughs = get_tec_dataset(score_dir, threshold)
    tec[~troughs] = np.inf
    min_tec = np.min(tec, axis=1)
    min_mlat = config.mlat_vals[np.argmin(tec, axis=1)]
    kp = io.get_kp(times)
    mask = ((config.mlt_vals > -5) & (config.mlt_vals < 5))[None, :] & np.isfinite(min_tec)

    y = np.any(troughs[:, :, (config.mlt_vals > -5) & (config.mlt_vals < 5)], axis=1)
    x = np.broadcast_to(kp[:, None], y.shape)
    mean_stat = binned_statistic(x.ravel(), y.ravel(), 'mean', np.arange(10))
    ax[0].bar(np.arange(9) + .7, mean_stat.statistic, .4, color=colors[1])

    x = np.broadcast_to(kp[:, None], min_mlat.shape)[mask]
    y = min_mlat[mask]
    mean_stat = binned_statistic(x, y, 'mean', np.arange(10))
    std_stat = binned_statistic(x, y, 'std', np.arange(10))
    reg = linregress(x, y)
    xr = np.array([0, 9])
    yr = reg.slope * xr + reg.intercept
    ax[1].plot(xr, yr, '-', c=colors[1], label='Ours')
    ax[1].errorbar(np.arange(9) + .6, mean_stat.statistic, yerr=std_stat.statistic, fmt='o', c=colors[1])

    y = np.sum(troughs[:, :, (config.mlt_vals > -5) & (config.mlt_vals < 5)], axis=1)
    x = np.broadcast_to(kp[:, None], y.shape)
    mask = y > 0
    mean_stat = binned_statistic(x[mask], y[mask], 'mean', np.arange(10))
    std_stat = binned_statistic(x[mask], y[mask], 'std', np.arange(10))
    ax[2].errorbar(np.arange(9) + .6, mean_stat.statistic, yerr=std_stat.statistic, fmt='o-', c=colors[1])


def main():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    plot_tec(ax, pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 1.5)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    plot_tec(ax, pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 2.0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    plot_tec(ax, pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 2.5)
    plot_swarm(ax)
    plt.show()


if __name__ == "__main__":
    main()
