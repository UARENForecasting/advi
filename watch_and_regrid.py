#! /usr/bin/env python
"""
Monitors directories to regrid any new datasets
"""
import argparse
import logging
from pathlib import Path
import time
import shutil


import pandas as pd


from regrid_wrf import FILENAME, read_subset, regrid_and_save


PREFIX = 'WRF'


def regrid_file(model, base_dir, day, variables, save_dir):
    errors = 0
    for var in variables:
        try:
            data, lats, lons, times, valid_date = read_subset(model,
                                                              base_dir,
                                                              day, var)
        except KeyError:
            logging.error('Failed to find variable %s in %s', var, model)
            errors += 1
            continue

        try:
            regrid_and_save(data, lats, lons, times, valid_date,
                            False, save_dir, var, model)
        except FileExistsError:
            logging.warning('Will not overwrite %s for %s', var, model)
            errors += 1
            continue
    return errors


def list_missing_variables(base_dir, save_dir, variables, days):
    bdp = Path(base_dir).expanduser()
    strfformat = f'%Y/%m/%d/{PREFIX}*/{FILENAME}'
    to_get = []
    for day in days:
        possible_model = [pp.parent.relative_to(bdp)
                          for pp in bdp.glob(day.strftime(strfformat))]
        for model in possible_model:
            model_save_dir = (save_dir / model).expanduser()
            if not model_save_dir.is_dir():
                logging.debug('Need all variables for %s', model)
                to_get.append((model, variables))
            else:
                missing_vars = []
                for var in variables:
                    if not (model_save_dir / f'{var}.h5').exists():
                        missing_vars.append(var)
                        logging.debug('%s missing for %s', var, model)
                if missing_vars:
                    to_get.append((model, missing_vars))
    return to_get


def loop_func(base_dir, save_dir, variables, max_days):
    today = pd.Timestamp.now(tz='UTC').date()
    days = pd.date_range(start=today - pd.Timedelta(days=max_days),
                         end=today, freq='1d')

    model_vars = list_missing_variables(base_dir, save_dir, variables,
                                        days)
    if not model_vars:
        logging.info('No models/variables to regrid')

    for model_dir, mvars in model_vars:
        day = pd.Timestamp('-'.join(model_dir.parts[:-1])).date()
        model = model_dir.parts[-1]
        logging.info('Regridding %s on %s for vars %s', model, day, mvars)
        regrid_file(model, base_dir, day, mvars, save_dir)

    all_dirs = {pp.parent for pp in Path(save_dir).rglob('WRF*')}

    keep_dirs = {Path(save_dir) / day.strftime('%Y/%m/%d')
                 for day in days}
    dirs_to_rm = all_dirs - keep_dirs
    for adir in dirs_to_rm:
        logging.info('Removing %s', adir)
        shutil.rmtree(adir)


def main():
    logging.basicConfig(
        level='WARNING',
        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    argparser = argparse.ArgumentParser(
        description='Monitor folders to regrid WRF data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-v', '--verbose', action='count',
                           help='Increase logging verbosity')
    argparser.add_argument('--save-dir', help='Directory to save data to',
                           default='~/.wrf')
    argparser.add_argument('--base-dir', help='Base directory with WRF files',
                           default='/a4/uaren/')
    argparser.add_argument('--max-days', type=int, default=3,
                           help='Maximum number of days to preserve data')
    argparser.add_argument('--polling-interval', default=30, type=int,
                           help='Period in seconds to poll base-dir')
    argparser.add_argument('vars', help='Variable to get from WRF file',
                           nargs='+')

    args = argparser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    while True:
        try:
            loop_func(args.base_dir, args.save_dir, args.vars, args.max_days)
            time.sleep(args.polling_interval)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
