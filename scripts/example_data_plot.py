import numpy as np
import h5py
from apexpy import Apex

import trough
from trough_stats.plotting import plot_polar_scatter, plot_polar_pcolormesh


def main():
    los_fn = "C:\\Users\\Greg\\Downloads\\los_20140219.001.h5"
    grid_fn = "C:\\Users\\Greg\\Downloads\\gps140219g.002.hdf5"
    rx_fn = "C:\\Users\\Greg\\Downloads\\site_20140219.001.h5"

    data = trough.get_data(np.datetime64('2014-02-19'), np.datetime64('2014-02-20'))
    data = data['tec'].sel(time='2014-02-19T07:00:00')
    proc_fig, _ = plot_polar_pcolormesh(data, figsize=(10, 6), publication=True, cmap='Blues_r', vmin=0, vmax=25)

    apex = Apex(date=2014)

    with h5py.File(grid_fn, 'r') as f:
        data = f['Data/Table Layout'][()]
    data = data[(data['hour'] == 7) & (data['gdlat'] > 0)]
    mlat, mlt = apex.convert(data['gdlat'], data['glon'], 'geo', 'mlt', 350, data['ut1_unix'])
    mask = mlat >= 30
    mad_fig, _ = plot_polar_scatter(mlat[mask], mlt[mask], data['tec'][mask], size=8, vmin=0, vmax=25, alpha=1,
                                    cmap='Blues_r', figsize=(6, 6), publication=True)

    with h5py.File(los_fn, 'r') as f:
        data = f['Data/Table Layout'][()]
    data = data[(data['hour'] == 7) & (data['elm'] > 15) & (data['gdlat'] > 0)]
    mlat, mlt = apex.convert(data['gdlat'], data['glon'], 'geo', 'mlt', 350, data['ut1_unix'])
    mask = mlat >= 30
    los_fig, _ = plot_polar_scatter(mlat[mask], mlt[mask], data['tec'][mask], size=4, vmin=0, vmax=25, alpha=.75,
                                    cmap='Blues_r', figsize=(6, 6), publication=True)

    for fig, name in zip([proc_fig, mad_fig, los_fig], ['processed', 'mad', 'los']):
        fn = f"C:\\Users\\Greg\\Desktop\\paper figs\\components\\{name}.png"
        fig.savefig(fn, dpi=300)


if __name__ == "__main__":
    main()
