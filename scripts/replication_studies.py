import pandas
import pathlib
import numpy as np
from scipy.stats import binned_statistic, linregress
from matplotlib import pyplot as plt

from ttools import io, utils, config, plotting

from get_dataset import get_tec_dataset

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use('default')
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


def aa_figure_8_swarm(ax):
    swarm_troughs = get_swarm_troughs(-.2)
    swarm_troughs = swarm_troughs[(swarm_troughs.min_mlt > -5) & (swarm_troughs.min_mlt < 5)]
    swarm_kp = io.get_kp(swarm_troughs.tec_time.values)
    yes_mask = swarm_troughs.trough
    min_mlat = swarm_troughs.min_mlat[yes_mask]

    mean_stat = binned_statistic(swarm_kp, yes_mask, 'mean', np.arange(10))
    count_stat = binned_statistic(swarm_kp, yes_mask, 'count', np.arange(10))
    s = np.sqrt(mean_stat.statistic * (1 - mean_stat.statistic) / (count_stat.statistic - 1))
    ax[0].bar(np.arange(9) + .3, mean_stat.statistic, .4, color=colors[0])
    ax[0].errorbar(np.arange(9) + .3, mean_stat.statistic, yerr=s, fmt='.', c='k', ms=0)

    mean_stat = binned_statistic(swarm_kp[yes_mask], min_mlat, 'mean', np.arange(10))
    std_stat = binned_statistic(swarm_kp[yes_mask], min_mlat, 'std', np.arange(10))
    reg = linregress(swarm_kp[yes_mask], min_mlat)
    x = np.array([0, 9])
    y = reg.slope * x + reg.intercept
    ax[1].plot(x, y, '-', c=colors[0], label='Aa 2020')
    ax[1].errorbar(np.arange(9) + .4, mean_stat.statistic, yerr=std_stat.statistic, fmt='o', c=colors[0])


def aa_figure_8_tec(ax, times, tec, troughs):
    fin = np.isfinite(tec)
    tec[~troughs] = np.inf
    min_tec = np.min(tec, axis=1)
    min_mlat = config.mlat_vals[np.argmin(tec, axis=1)]
    kp = io.get_kp(times)
    mask = ((config.mlt_vals > -5) & (config.mlt_vals < 5))[None, :] & np.isfinite(min_tec)

    y = np.any(troughs[:, :, ((config.mlt_vals > -5) & (config.mlt_vals < 5))], axis=1)
    f = np.mean(fin[:, :, ((config.mlt_vals > -5) & (config.mlt_vals < 5))], axis=1) >= .5
    x = np.broadcast_to(kp[:, None], y.shape)
    mean_stat = binned_statistic(x[f], y[f], 'mean', np.arange(10))
    count_stat = binned_statistic(x[f], y[f], 'count', np.arange(10))
    s = np.sqrt(mean_stat.statistic * (1 - mean_stat.statistic) / (count_stat.statistic - 1))
    ax[0].bar(np.arange(9) + .7, mean_stat.statistic, .4, color=colors[1])
    ax[0].errorbar(np.arange(9) + .7, mean_stat.statistic, yerr=s, fmt='.', c='k', ms=0)

    x = np.broadcast_to(kp[:, None], min_mlat.shape)[mask]
    y = min_mlat[mask]
    mean_stat = binned_statistic(x, y, 'mean', np.arange(10))
    std_stat = binned_statistic(x, y, 'std', np.arange(10))
    reg = linregress(x, y)
    xr = np.array([0, 9])
    yr = reg.slope * xr + reg.intercept
    ax[1].plot(xr, yr, '-', c=colors[1], label='Ours')
    ax[1].errorbar(np.arange(9) + .6, mean_stat.statistic, yerr=std_stat.statistic, fmt='o', c=colors[1])


def aa_figure_2ghi_swarm(ax):
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


def aa_figure_2ghi_tec(ax, times, tec, troughs):
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


def aa_figure_4a(times, tec, troughs):
    times = times.copy()
    tec = tec.copy()
    troughs = troughs.copy()
    kp = io.get_kp(times)
    times = times[kp <= 3]
    tec_troughs = troughs[kp <= 3]
    tec = tec[kp <= 3]

    x = (times.astype('datetime64[s]') - times.astype('datetime64[Y]')).astype('timedelta64[s]').astype(float) / (60 * 60 * 24)
    x = np.broadcast_to(x[:, None], (tec_troughs.shape[0], tec_troughs.shape[2]))
    y = np.broadcast_to(config.mlt_vals[None, :], (tec_troughs.shape[0], tec_troughs.shape[2]))
    y = y + np.random.randn(*y.shape) * .02

    trough_mask = np.any((tec_troughs * np.isfinite(tec)), axis=1)
    obs_mask = np.any(np.isfinite(tec), axis=1)

    total_counts, *_ = np.histogram2d(x[obs_mask], y[obs_mask], bins=[40, 40], range=[(0, 365), [-12, 12]])

    fig, ax = plt.subplots(dpi=300)
    counts, xe, ye, pcm = ax.hist2d(x[trough_mask], y[trough_mask], bins=[40, 40], range=[(0, 365), [-12, 12]], cmap='jet')

    fig, ax = plt.subplots(dpi=300)
    prob = counts / total_counts
    prob[total_counts < 100] = np.nan
    pcm = ax.pcolormesh(xe, ye, prob.T, cmap='jet')
    l = np.datetime64('2010-01-01T00:00:00') + np.arange(6).astype('timedelta64[M]').astype('timedelta64[s]') * 2
    l = (l.astype('datetime64[s]') - l.astype('datetime64[Y]')).astype('timedelta64[s]').astype(float) / (60 * 60 * 24)
    ax.set_xticks(l)
    plt.colorbar(pcm)


def aa_figure_2abc(times, tec, troughs):
    times = times.copy()
    tec = tec.copy()
    troughs = troughs.copy()
    kp = io.get_kp(times)
    times = times[kp <= 3]
    tec_troughs = troughs[kp <= 3]
    tec = tec[kp <= 3]
    fin = np.isfinite(tec)

    trough = tec_troughs & fin

    months = (times.astype('datetime64[M]') - times.astype('datetime64[Y]')).astype(int)

    for i, (season, mo) in enumerate(MONTH_INDEX.items()):
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(polar=True)
        ax.set_title(season)
        mask = np.zeros_like(months, dtype=bool)
        for m in mo:
            mask |= months == m
        trough_sum = np.sum(trough[mask], axis=0)
        all_sum = np.sum(fin[mask], axis=0)
        p = trough_sum / all_sum
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, p, cmap='jet', vmin=0)
        plt.colorbar(pcm)
        plotting.format_polar_mag_ax(ax)


def aa_figure_2ghi(times, tec, troughs):
    plt.style.use('ggplot')
    times = times.copy()
    tec = tec.copy()
    troughs = troughs.copy()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    aa_figure_2ghi_swarm(ax)
    aa_figure_2ghi_tec(ax, times, tec, troughs)
    plt.style.use('default')


def aa_figure_8(times, tec, troughs):
    plt.style.use('ggplot')
    times = times.copy()
    tec = tec.copy()
    troughs = troughs.copy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    aa_figure_8_swarm(ax)
    aa_figure_8_tec(ax, times, tec, troughs)
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(40, 80)
    plt.style.use('default')


if __name__ == "__main__":
    score_dir = pathlib.Path("E:\\ttools\\tec\\score\\l2_3")
    for threshold in [1.5, 2.0, 2.5]:
        times, tec, troughs = get_tec_dataset(score_dir, threshold)
        aa_figure_2abc(times, tec, troughs)
        aa_figure_2ghi(times, tec, troughs)
        aa_figure_4a(times, tec, troughs)
        aa_figure_8(times, tec, troughs)

    score_dir = pathlib.Path("E:\\ttools\\tec\\score\\l2_9")
    for threshold in [0.5, 1.0, 1.5]:
        times, tec, troughs = get_tec_dataset(score_dir, threshold)
        aa_figure_2abc(times, tec, troughs)
        aa_figure_2ghi(times, tec, troughs)
        aa_figure_4a(times, tec, troughs)
        aa_figure_8(times, tec, troughs)

    plt.show()
