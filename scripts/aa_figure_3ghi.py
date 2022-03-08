import pandas
import pathlib
import numpy as np
from scipy.stats import binned_statistic, linregress
from matplotlib import pyplot as plt

from ttools import io, utils, config

from get_dataset import get_tec_dataset


plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
MONTH_INDEX = {
    'winter': [0, 1, 10, 11],
    'equinox': [2, 3, 8, 9],
    'summer': [4, 5, 6, 7],
}


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
    swarm_troughs = swarm_troughs[swarm_troughs.trough]
    swarm_kp = io.get_kp(swarm_troughs.tec_time.values)
    swarm_troughs = swarm_troughs[swarm_kp <= 3]
    x = swarm_troughs.min_mlt
    y = swarm_troughs.min_mlat
    time = swarm_troughs.tec_time.values.astype('datetime64[s]')
    months = (time.astype('datetime64[M]') - time.astype('datetime64[Y]')).astype(int)
    be = np.arange(-12, 14) - .5
    bc = np.arange(-12, 13)

    for i, (season, mo) in enumerate(MONTH_INDEX.items()):
        mask = np.zeros_like(months, dtype=bool)
        for m in mo:
            mask |= months == m
        mean_result = binned_statistic(x[mask], y[mask], 'mean', be)
        std_result = binned_statistic(x[mask], y[mask], 'std', be)
        ax[i].errorbar(bc - .2, mean_result.statistic, yerr=std_result.statistic, fmt='-', c=colors[0], errorevery=2)
        ax[i].set_title(season)


def plot_tec(ax, score_dir, threshold):
    times, tec, troughs = get_tec_dataset(score_dir, threshold)
    tec[~troughs] = np.inf
    min_tec = np.min(tec, axis=1)
    kp = io.get_kp(times)
    mask = np.isfinite(min_tec) & (kp <= 3)[:, None]
    x = np.broadcast_to(config.mlt_vals[None, :], mask.shape)[mask]
    t = np.broadcast_to(times[:, None], mask.shape)[mask].astype('datetime64[s]')
    y = config.mlat_vals[np.argmin(tec, axis=1)]
    y = y[mask]

    months = (t.astype('datetime64[M]') - t.astype('datetime64[Y]')).astype(int)
    be = np.arange(-12, 14) - .5
    bc = np.arange(-12, 13)
    for i, (season, mo) in enumerate(MONTH_INDEX.items()):
        mask = np.zeros_like(months, dtype=bool)
        for m in mo:
            mask |= months == m
        mean_result = binned_statistic(x[mask], y[mask], 'mean', be)
        std_result = binned_statistic(x[mask], y[mask], 'std', be)
        ax[i].errorbar(bc + .2, mean_result.statistic, yerr=std_result.statistic, fmt='-', c=colors[1], errorevery=2)


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
