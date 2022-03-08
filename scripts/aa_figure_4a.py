import numpy as np
import pathlib
import h5py
import matplotlib.pyplot as plt

from ttools import config, io

from get_dataset import get_tec_dataset


def plot_season_mlt_probability(score_dir, threshold):
    times, tec, troughs = get_tec_dataset(score_dir, threshold)
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

    plt.figure()
    plt.colorbar(pcm)

    plt.show()


if __name__ == "__main__":
    plot_season_mlt_probability(pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 1.5)
    plot_season_mlt_probability(pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 2.0)
    plot_season_mlt_probability(pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 2.5)
