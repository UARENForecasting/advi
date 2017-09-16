# coding: utf-8
import os
import sys


try:
    _var = sys.argv[1]
except IndexError:
    _var = 'temp'


ALPHA = 0.7
RED = '#AB0520'
BLUE = '#0C234B'
TITLE = 'UA HAS ADVI'
DATA_DIRECTORY = os.getenv('ADVI_DATADIR', '~/.wrf')
WS_ORIGIN = os.getenv('WS_ORIGIN', 'localhost:5006')
PREFIX = os.getenv('ADVI_PREFIX', '')
POSSIBLE_MODELS = ('WRFGFS_00Z', 'WRFGFS_06Z', 'WRFGFS_12Z',
                   'WRFNAM_00Z', 'WRFNAM_06Z', 'WRFNAM_12Z')

MENU_VARS = (('2m Temperature', 'temp'),
             ('1 hr Temperature Change', 'dt'),
             ('10m Wind Speed', 'wspd'),
             ('1 hr Precip', 'rain'),
             ('Accumulated Precip', 'rainac'),
             ('GHI', 'ghi'),
             ('DNI', 'dni'))


if _var == 'radar':
    MIN_VAL = -80
    MAX_VAL = 80
    VAR = 'REFD_MAX'
    CMAP = 'plasma'
    XLABEL = 'Max Radar Refl. (dbZ)'
    NBINS = 25
elif _var == 'rain':
    MIN_VAL = 0
    MAX_VAL = 2
    VAR = 'RAIN1H'
    CMAP = 'magma'
    XLABEL = 'One-hour Precip (in)'
    NBINS = 21
elif _var == 'rainac':
    MIN_VAL = 0
    MAX_VAL = 2
    VAR = 'RAINNC'
    CMAP = 'magma'
    XLABEL = 'Precip Accumulation (in)'
    NBINS = 21
elif _var == 'dt':
    MIN_VAL = -20
    MAX_VAL = 20
    VAR = 'DT'
    CMAP = 'coolwarm'
    XLABEL = 'One-Hour Temperature Change (°F)'
    NBINS = 40
elif _var == 'temp':
    MIN_VAL = 0
    MAX_VAL = 120
    VAR = 'T2'
    CMAP = 'plasma'
    XLABEL = '2m Temperature (°F)'
    NBINS = 61
elif _var == 'wspd':
    MIN_VAL = 0
    MAX_VAL = 44
    VAR = 'WSPD'
    CMAP = 'viridis'
    XLABEL = '10m Wind Speed (knots)'
    NBINS = 25
elif _var == 'ghi':
    MIN_VAL = 0
    MAX_VAL = 1200
    VAR = 'SWDNB'
    CMAP = 'viridis'
    XLABEL = 'GHI (W/m^2)'
    NBINS = 25
elif _var == 'dni':
    MIN_VAL = 0
    MAX_VAL = 1100
    VAR = 'SWDDNI'
    CMAP = 'viridis'
    XLABEL = 'DNI (W/m^2)'
    NBINS = 25
