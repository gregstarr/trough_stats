import numpy as np
from datetime import datetime
from matplotlib.pyplot import show

import trough
from trough_stats import computation, plotting


def main():
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2020, 1, 1)

    data = trough.get_data(start_date, end_date)
    min_mlat = computation.get_min_mlat(data['tec'], data['labels'])

    kp_mask = data['kp'] <= 30
    season_occ_rate = data['labels'][kp_mask].groupby('time.season').mean('time')
    plotting.plot_polar_pcolormesh(season_occ_rate)

    mlt_bins = np.linspace(-8, 8, 12)
    kp_mlat_result = min_mlat.groupby_bins(data['kp'], 9).mean('time').groupby_bins('mlt', mlt_bins).mean()

    plotting.plot_polar_lines(kp_mlat_result, 'Blues')

    show()


if __name__ == "__main__":
    main()
