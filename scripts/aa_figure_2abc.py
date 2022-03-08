import numpy as np
import pathlib
import h5py
import matplotlib.pyplot as plt

from ttools import config, io, plotting

from get_dataset import get_tec_dataset


MONTH_INDEX = {
    'winter': [0, 1, 10, 11],
    'equinox': [2, 3, 8, 9],
    'summer': [4, 5, 6, 7],
}


def plot_season_mlt_probability(score_dir, threshold):
    times, tec, troughs = get_tec_dataset(score_dir, threshold)
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
        plotting.format_polar_mag_ax(ax)
        plt.colorbar(pcm)


if __name__ == "__main__":
    plot_season_mlt_probability(pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 1.5)
    plot_season_mlt_probability(pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 2.0)
    plot_season_mlt_probability(pathlib.Path("E:\\ttools\\tec\\score\\l2_3"), 2.5)
    plt.show()
