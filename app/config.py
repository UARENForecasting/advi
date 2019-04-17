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
GA_TRACKING_ID = os.getenv('ADVI_TRACKING_ID', '')
PREFIX = os.getenv('ADVI_PREFIX', '')
POSSIBLE_MODELS = ('WRFGFS_00Z', 'WRFGFS_06Z', 'WRFGFS_12Z',
                   'WRFNAM_00Z', 'WRFNAM_06Z', 'WRFNAM_12Z', 'WRFNAM_18Z',
                   'WRFRUC_09Z', 'WRFRUC_12Z', 'WRFRUC_18Z')
CUSTOM_BOKEH_MODELS = (('app.models.binned_color_mapper', 'BinnedColorMapper'),
                       ('app.models.disabled_select', 'DisabledSelect'))
ANIMATE_TIME = 500

MENU_VARS = (('2m Temperature', 'temp'),
             ('1 hr Temperature Change', 'dt'),
             ('10m Wind Speed', 'wspd'),
             ('1 hr Precip', 'rain'),
             ('Accumulated Precip', 'rainac'),
             ('GHI', 'ghi'),
             ('DNI', 'dni'),
             ('Composite Radar', 'radar'),
             ('AOD 550', 'aod550'))

LEVELS = []
if _var == 'radar':
    MIN_VAL = 0
    MAX_VAL = 80
    VAR = 'MDBZ'
    CMAP = 'nws_radar'
    XLABEL = 'Composite Reflectivity (dBZ)'
    NBINS = 17
elif _var == 'rain':
    MIN_VAL = 0
    MAX_VAL = 2
    VAR = 'RAIN1H'
    CMAP = 'precip_1h'
    XLABEL = 'One-hour Precip (in)'
    LEVELS = [
        0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3
    ]
elif _var == 'rainac':
    MIN_VAL = 0
    MAX_VAL = 10
    VAR = 'RAINNC'
    CMAP = 'precip_accum'
    XLABEL = 'Precip Accumulation (in)'
    LEVELS = [
        0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11, 12
    ]
elif _var == 'dt':
    MIN_VAL = -20
    MAX_VAL = 20
    VAR = 'DT'
    CMAP = 'coolwarm'
    XLABEL = 'One-Hour Temperature Change (°F)'
    NBINS = 41
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
elif _var == 'aod550':
    MIN_VAL = 0
    MAX_VAL = .3
    VAR = 'AOD550'
    CMAP = 'plasma'
    XLABEL = 'AOD 550 nm'
    NBINS = 25
