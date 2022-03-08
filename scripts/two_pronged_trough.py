import numpy as np
from matplotlib import pyplot as plt

import trough
from trough_stats.plotting import plot_polar_pcolormesh


def main():
    data = trough.get_data(np.datetime64('2018-01-01'), np.datetime64('2019-12-31'))
    mlats = data['mlat'].where(data['labels'])
    width_r = mlats.max(dim='mlat', skipna=True) - mlats.min(dim='mlat', skipna=True) + 1
    width_s = data['labels'].sum(dim='mlat')
    mask = (width_r > width_s + 5) & (width_s > 2)
    idx = np.argwhere(mask.values)
    i = np.unique(idx[:, 0])[14]
    plot_polar_pcolormesh(data['tec'][i])
    plot_polar_pcolormesh(data['labels'][i])
    print(data['time'][i])
    plt.show()


if __name__ == "__main__":
    main()
