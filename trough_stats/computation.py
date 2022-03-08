import numpy as np
import xarray as xr


def get_min_mlat(tec, labels):
    masked_tec = tec.values.copy()
    masked_tec[labels.values == 0] = np.inf
    min_mlat = tec.mlat.values[np.argmin(masked_tec, axis=1)]
    min_mlat[~np.any(np.isfinite(masked_tec), axis=1)] = np.nan
    return xr.DataArray(
        min_mlat,
        coords={
            'time': tec.time,
            'mlt': tec.mlt
        },
        dims=['time', 'mlt']
    )
