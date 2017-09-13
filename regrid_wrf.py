#! /usr/bin/env python
"""
Download the MRMS data, get only the given area, project to web mercator,
and regrid onto a regular grid for later plotting.
"""
import argparse
import datetime as dt
from functools import partial
import logging
import os
import sys
import warnings


import pandas as pd
import numpy as np
import netCDF4 as nc4
import scipy.interpolate
import tables


FILENAME = 'wrfsolar_d02_hourly.nc'


def webmerc_proj(lat, lon):
    """Convert latititude and longitude to web mercator"""
    R = 6378137
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def convert_time_str(time_bytes):
    time_str = time_bytes.astype(str)
    times = [''.join(ts).replace('_', ' ') for ts in time_str]
    indx = pd.DatetimeIndex(times).tz_localize('UTC')
    return indx


def read_subset(model, base_dir, day, variable):
    """Read a subset of the data from the grib file"""
    logging.info('Reading subset of data from grib file')
    filename = os.path.join(os.path.expanduser(base_dir),
                            day.strftime('%Y/%m/%d'),
                            model,
                            FILENAME)
    if not os.path.isfile(filename):
        logging.error('File %s does not exist', filename)
        sys.exit(1)
    ds = nc4.Dataset(filename)
    lats = ds.variables['XLAT'][:]
    lons = ds.variables['XLONG'][:]
    try:
        data = ds.variables[variable][:]
    except KeyError:
        raise
    times = convert_time_str(ds.variables['Times'][:])
    valid_date = pd.Timestamp(ds.SIMULATION_START_DATE.replace('_', ' '))
    return data, lats, lons, times, valid_date


def regrid_and_save(data, lats, lons, times, valid_date, overwrite, save_dir,
                    var, model):
    """Regrid the data onto an even web mercator grid"""
    logging.info('Regridding data...')
    x, y = webmerc_proj(lats, lons)

    h5file = create_file(save_dir, valid_date, model, f'{var}.h5', overwrite)
    save = partial(save_data, h5file)

    shape = data.shape
    # make new grid
    xn = np.linspace(x.min(), x.max(), shape[2])
    yn = np.linspace(y.min(), y.max(), shape[1])
    X, Y = np.meshgrid(xn, yn)
    save({'X': X, 'Y': Y})

    lni = scipy.interpolate.LinearNDInterpolator(
        (x.ravel(), y.ravel()),
        data[0].ravel())
    dtri = lni.tri
    regridded = lni((X, Y))
    tformat = '%Y%m%dT%H%MZ'
    save({times[0].strftime(tformat): regridded})
    for i in range(1, data.shape[0]):
        lni = scipy.interpolate.LinearNDInterpolator(
            dtri, data[i].ravel())
        regridded = lni((X, Y))
        save({times[i].strftime(tformat): regridded})
    h5file.create_array('/', 'times', times.values.astype(int))
    h5file.root._v_attrs.valid_date = valid_date
    h5file.close()


def create_file(base_dir, valid_date, model, filename, overwrite):
    thedir = os.path.join(os.path.expanduser(base_dir),
                          valid_date.strftime('%Y/%m/%d'),
                          model)
    if not os.path.isdir(thedir):
        os.makedirs(thedir)

    path = os.path.join(thedir, filename)
    logging.info('Creating h5 file at %s', path)
    if os.path.isfile(path) and not overwrite:
        logging.error('%s already exists', path)
        sys.exit(1)
    f = tables.Filters(fletcher32=True, shuffle=True, complib='blosc:zlib',
                       complevel=5)
    h5file = tables.open_file(path, mode='w', filters=f)
    return h5file


def save_data(h5file, ddict):
    """Save the data and grid to a numpy file"""
    logging.info('Saving numpy data to a file...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k, v in ddict.items():
            h5file.create_array('/', k, v)
    h5file.flush()


def main():
    logging.basicConfig(
        level='WARNING',
        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    argparser = argparse.ArgumentParser(
        description='Regrid and save WRF data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-v', '--verbose', action='count',
                           help='Increase logging verbosity')
    argparser.add_argument('--save-dir', help='Directory to save data to',
                           default='~/.wrf')
    argparser.add_argument('-o', '--overwrite', action='store_true',
                           help='Overwrite file if already exists')
    argparser.add_argument('--var', help='Variable to get from WRF file',
                           default='T2')
    argparser.add_argument('--base-dir', help='Base directory with WRF files',
                           default='/a4/uaren/')
    argparser.add_argument('-d', '--day', help='Day to get data from')
    argparser.add_argument(
        'model', help='Model to get data from i.e. WRFGFS_12Z')

    args = argparser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.day is None:
        day = dt.date.today()
    else:
        day = pd.Timestamp(args.day).date()

    data, lats, lons, times, valid_date = read_subset(args.model,
                                                      args.base_dir,
                                                      day, args.var)
    regrid_and_save(data, lats, lons, times, valid_date, args.overwrite,
                    args.save_dir, args.var, args.model)


if __name__ == '__main__':
    main()
