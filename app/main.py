# coding: utf-8

from collections import OrderedDict
import datetime as dt
from functools import partial
import importlib
import logging
from pathlib import Path
import os
import sys
import warnings


from bokeh import events
from bokeh.colors import RGB
from bokeh.models import (
    Range1d, LinearColorMapper, ColorBar, FixedTicker,
    ColumnDataSource, WMTSTileSource, Slider)
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Select, Div, RadioButtonGroup
from bokeh.server.callbacks import PeriodicCallback
from bokeh.plotting import figure, curdoc
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable, get_cmap, register_cmap
import numpy as np
import pandas as pd
import tables
from tornado import gen


from models.disabled_select import DisabledSelect
from models.binned_color_mapper import BinnedColorMapper
import config
# reload since sys.argv changes values
config = importlib.reload(config)

nws_radar_cmap = ListedColormap(
    name='nws_radar', colors=(
        "#646464",
        "#04e9e7",
        "#019ff4",
        "#0300f4",
        "#02fd02",
        "#01c501",
        "#008e00",
        "#fdf802",
        "#e5bc00",
        "#fd9500",
        "#fd0000",
        "#d40000",
        "#bc0000",
        "#f800fd",
        "#9854c6",
        "#fdfdfd",
        ))
precip_1h_cmap = ListedColormap(
    name='precip_1h', colors=(
        '#ffffff',
        '#ebf6ff',
        '#d6e2ff',
        '#8eb2ff',
        '#7f96ff',
        '#7285f8',
        '#6370f8',
        '#009e1e',
        '#b3d16e',
        '#fff913',
        '#ffa309',
        '#e50000',
        '#bd0000'
    ))
precip_accum_cmap = ListedColormap(
    name='precip_accum', colors=(
        '#ffffff',
        '#ebf6ff',
        '#d6e2ff',
        '#8eb2ff',
        '#7f96ff',
        '#7285f8',
        '#6370f8',
        '#009e1e',
        '#b3d16e',
        '#fff913',
        '#ffa309',
        '#e50000',
        '#bd0000',
        '#810000',
        '#000000',
        '#767676',
        '#aaaaaa',
        '#cdcdcd',
        '#eeeeee',
        '#e1b4fa',
        '#950fdf'
    ))
register_cmap(cmap=nws_radar_cmap)
register_cmap(cmap=precip_1h_cmap)
register_cmap(cmap=precip_accum_cmap)


class H5File(object):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.t = tables.open_file(self.path, mode='r')
        return self.t

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.t.close()
        except:
            pass


def load_file(model, fx_date='latest'):
    dir = os.path.expanduser(config.DATA_DIRECTORY)
    if fx_date == 'latest':
        p = Path(dir)
        model_dir = sorted([pp for pp in p.rglob(f'*{model}')],
                           reverse=True)[0]
    else:
        model_dir = os.path.join(dir, fx_date.strftime('%Y/%m/%d'),
                                 strpmodel(model))

    path = os.path.join(model_dir,
                        f'{config.VAR}.h5')

    global h5file
    h5file = H5File(path)
    global times
    with h5file as h5:
        times = pd.DatetimeIndex(
            h5.get_node('/times')[:]).tz_localize('UTC')


def load_data(valid_date):
    strformat = '%Y%m%dT%H%MZ'
    with h5file as h5:
        regridded_data = h5.get_node(f'/{valid_date.strftime(strformat)}')[:]
        regridded_data[np.isnan(regridded_data)] = -999

        X = h5.get_node('/X')[:]
        Y = h5.get_node('/Y')[:]
    masked_regrid = np.ma.masked_less(regridded_data, -998)
    return masked_regrid, X, Y


def load_tseries(xi, yi):
    strformat = '%Y%m%dT%H%MZ'
    rd = []
    with h5file as h5:
        for t in times:
            rd.append(h5.get_node(f'/{t.strftime(strformat)}')[yi, xi])
    data = pd.Series(rd, index=times)
    return data


def find_fx_times():
    p = Path(config.DATA_DIRECTORY).expanduser()
    out = OrderedDict()
    for pp in sorted(p.rglob(f'*WRF*')):
        try:
            datetime = dt.datetime.strptime(''.join(pp.parts[-4:-1]),
                                            '%Y%m%d')
        except ValueError:
            logging.debug('%s does not conform to expected format', pp)
            continue
        if not pp.joinpath(f'{config.VAR}.h5').exists():
            logging.debug('No h5 file for %s in %s', config.VAR, pp)
            continue
        date = datetime.strftime('%Y-%m-%d')
        out[date] = datetime
    return out


def strfmodel(modelstr):
    return f'{modelstr[3:6]} {modelstr[7:]}'


def strpmodel(model):
    m = model.split(' ')
    return f'WRF{m[0]}_{m[1]}'


def get_models(date):
    dir = os.path.join(config.DATA_DIRECTORY, date.strftime('%Y/%m/%d'))
    p = Path(dir).expanduser()
    disabled = {model: True for model in config.POSSIBLE_MODELS}
    for pp in p.iterdir():
        if pp.joinpath(f'{config.VAR}.h5').exists():
            m = pp.parts[-1]
            disabled[m] = False
    mld = [(strfmodel(k), v) for k, v in disabled.items()]
    return mld


# setup the coloring
if config.LEVELS:
    levels = np.array(config.LEVELS)
    ticks = list(range(len(levels)))
    tick_labels = {ticks[i]: '{:0.2f}'.format(levels[i])
                   for i in range(len(ticks))}
    cbkwargs = dict(major_label_overrides=tick_labels,
                    label_standoff=9)
    cmkwargs = dict(low=0, high=len(levels) - 1)
    bin_width = ticks[1] - ticks[0]
    bin_centers = np.array(ticks[:-1]) + bin_width / 2
    hist_xrange_kwargs = dict(start=min(ticks), end=max(ticks))
else:
    levels = MaxNLocator(nbins=config.NBINS).tick_values(
        config.MIN_VAL, config.MAX_VAL)
    ticks = levels[::3]
    cbkwargs = {}
    cmkwargs = dict(low=config.MIN_VAL, high=config.MAX_VAL)
    bin_width = levels[1] - levels[0]
    bin_centers = levels[:-1] + bin_width / 2
    hist_xrange_kwargs = dict(start=config.MIN_VAL, end=config.MAX_VAL)

cmap = get_cmap(config.CMAP)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
sm = ScalarMappable(norm=norm, cmap=cmap)
color_pal = [RGB(val[0], val[1], val[2], config.ALPHA) for val in
             sm.to_rgba(levels, bytes=True, norm=True)[:-1]]

bin_pal = color_pal.copy()
bin_pal.append(RGB(255, 255, 255, config.ALPHA))
bin_mapper = BinnedColorMapper(bin_pal)
color_mapper = LinearColorMapper(color_pal, **cmkwargs)
ticker = FixedTicker(ticks=ticks)
cb = ColorBar(color_mapper=color_mapper, location=(0, 0),
              ticker=ticker, **cbkwargs)

# make the bokeh figures without the data yet
width = 768
height = int(width / 1.6)

tools = 'pan, box_zoom, reset, save'
map_fig = figure(plot_width=width, plot_height=height,
                 y_axis_type=None, x_axis_type=None,
                 toolbar_location='left', tools=tools + ', wheel_zoom',
                 active_scroll='wheel_zoom',
                 toolbar_sticky=False, sizing_mode='scale_width',
                 title='', name='map_fig')


rgba_img_source = ColumnDataSource(data={'image': [], 'x': [], 'y': [],
                                         'dw': [], 'dh': []})
rgba_img = map_fig.image(image='image', x='x', y='y', dw='dw', dh='dh',
                         source=rgba_img_source, color_mapper=bin_mapper)


# Need to use this and not bokeh.tile_providers.STAMEN_TONER
# https://github.com/bokeh/bokeh/issues/4770
STAMEN_TONER = WMTSTileSource(
    url=(os.getenv('TILE_SOURCE',
                   'https://stamen-tiles.a.ssl.fastly.net/toner-lite') +
         '/{Z}/{X}/{Y}.png'),
    attribution=(
        'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
        'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0'
        '</a>. Map data by <a href="http://openstreetmap.org">OpenStreetMap'
        '</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
    )
)
map_fig.add_tile(STAMEN_TONER)
map_fig.add_layout(cb, 'right')

hheight = int(width / 2)
# Make the histogram figure
hist_fig = figure(plot_width=hheight, plot_height=hheight,
                  toolbar_location='right',
                  x_axis_label=config.XLABEL,
                  y_axis_label='Counts', tools=tools + ', ywheel_zoom',
                  active_scroll='ywheel_zoom', sizing_mode='scale_width',
                  x_range=Range1d(**hist_xrange_kwargs),
                  title='Histogram of map pixels',
                  name='hist_fig')

# make histograms
histbars = hist_fig.vbar(x=bin_centers, top=[3.0e6] * len(bin_centers),
                         width=bin_width, bottom=0,
                         color=color_pal)
hist_source = histbars.data_source

# line and point on map showing tapped location value
line_source = ColumnDataSource(data={'x': [-1, -1], 'y': [0, 1]})
if config.LEVELS:
    hist_fig.xaxis[0].major_label_overrides = tick_labels
else:
    hist_fig.line(x='x', y='y', color=config.RED, source=line_source,
                  alpha=config.ALPHA)
hover_pt = ColumnDataSource(data={'x': [0], 'y': [0], 'x_idx': [0],
                                  'y_idx': [0], 'lat': [0], 'lon': [0]})
map_fig.x(x='x', y='y', size=10, color=config.RED, alpha=config.ALPHA,
          source=hover_pt, level='overlay')

file_dict = find_fx_times()
dates = list(file_dict.keys())[::-1]
select_day = Select(title='Initialization Day', value=dates[0], options=dates,
                    sizing_mode='scale_width', name='select_day')
select_model = DisabledSelect(title='Initialization', value='',
                              options=[], sizing_mode='scale_width',
                              name='select_model')
times = []
select_fxtime = Slider(title='Forecast Time-Step', start=0, end=1, value=0,
                       name='timeslider', sizing_mode='scale_width')
play_buttons = RadioButtonGroup(labels=['\u25B6', '\u25FC', '\u27F3'],
                                active=1, name='play_buttons',
                                sizing_mode='fixed')
info_data = ColumnDataSource(data={'current_val': [0], 'mean': [0],
                                   'median': [0],
                                   'bin_width': [bin_width]})
info_text = """
<div class="well">
<b>Selected Value:</b> {current_val:0.1f} <b>Area Mean:</b> {mean:0.1f} <b>Latitude:</b> {lat:0.2f} <b>Longitude:</b> {lon:0.2f} <b>Bin Width:</b> {bin_width:0.1f}
</div>
"""  # NOQA
info_div = Div(sizing_mode='scale_width', name='info_div')

# Setup the updates for all the data
local_data_source = ColumnDataSource(data={'masked_regrid': [0], 'xn': [0],
                                           'yn': [0],
                                           'valid_date': [dt.datetime.now()]})

# timeseries plot
tseries_source = ColumnDataSource(
    data={'time': [0, 0],
          'values': [config.MAX_VAL, config.MAX_VAL]})
curpt_source = ColumnDataSource(data={'time': [0],
                                      'value': [config.MAX_VAL]})
tseries_fig = figure(
    height=hheight, width=hheight,
    x_axis_type='datetime',
    tools=tools + ', wheel_zoom',
    active_scroll='wheel_zoom',
    toolbar_location='left',
    title='Time-series at selected location',
    x_axis_label='Datetime',
    y_axis_label=config.XLABEL,
    name='tseries', sizing_mode='scale_width')
tseries_fig.line(x='time', y='values', source=tseries_source,
                 color=config.BLUE)
tseries_fig.diamond(x='time', y='value', color=config.RED, source=curpt_source,
                    level='overlay', size=6)


def update_histogram(attr, old, new):
    # makes it so only one callback added per 100 ms
    try:
        doc.add_timeout_callback(_update_histogram, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_histogram():
    left = map_fig.x_range.start
    right = map_fig.x_range.end
    bottom = map_fig.y_range.start
    top = map_fig.y_range.end

    masked_regrid = local_data_source.data['masked_regrid'][0]
    xn = local_data_source.data['xn'][0]
    yn = local_data_source.data['yn'][0]

    left_idx = np.abs(xn - left).argmin()
    right_idx = np.abs(xn - right).argmin() + 1
    bottom_idx = np.abs(yn - bottom).argmin()
    top_idx = np.abs(yn - top).argmin() + 1
    logging.debug('Updating histogram...')
    try:
        new_subset = masked_regrid[bottom_idx:top_idx, left_idx:right_idx]
    except TypeError:
        return
    counts, _ = np.histogram(
        new_subset.clip(max=config.MAX_VAL), bins=levels,
        range=(levels.min(), levels.max()))
    line_source.data.update({'y': np.array([0, counts.max()])})
    hist_source.data.update({'top': counts})
    logging.debug('Done updating histogram')

    info_data.data.update({'mean': np.array([float(new_subset.mean())])})
    try:
        doc.add_next_tick_callback(_update_div_text)
    except ValueError:
        pass


def update_map(attr, old, new):
    try:
        doc.add_timeout_callback(_update_histogram, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_map(update_range=False):
    logging.debug('Updating map...')
    valid_date = local_data_source.data['valid_date'][0]
    mfmt = '%Y-%m-%d %H:%M MST'
    map_fig.title.text = (
        f'UA HAS WRF {config.XLABEL} from {select_day.value} '
        f'{select_model.value} valid at '
        f'{valid_date.tz_convert("MST").strftime(mfmt)}')
    masked_regrid = local_data_source.data['masked_regrid'][0]
    xn = local_data_source.data['xn'][0]
    yn = local_data_source.data['yn'][0]
    dx = xn[1] - xn[0]
    dy = yn[1] - yn[0]
    vals = (np.digitize(masked_regrid.filled(np.inf), levels).astype('uint8')
            - 1)
    rgba_img_source.data.update({'image': [vals],
                                 'x': np.array([xn[0] - dx / 2]),
                                 'y': np.array([yn[0] - dy / 2]),
                                 'dw': np.array([xn[-1] - xn[0] + dx]),
                                 'dh': np.array([yn[-1] - yn[0] + dy])})
    if update_range:
        map_fig.x_range.start = xn[0]
        map_fig.x_range.end = xn[-1]
        map_fig.y_range.start = yn[0]
        map_fig.y_range.end = yn[-1]
    logging.debug('Done updating map')


def animate_times(attr, old, new):
    if new == 0:
        doc.add_periodic_callback(_alter_time_value, config.ANIMATE_TIME)
    elif new == 2:
        doc.add_next_tick_callback(_reset_time_value)
    else:
        for c in doc.session_callbacks:
            if isinstance(c, PeriodicCallback):
                doc.remove_periodic_callback(c)


@gen.coroutine
def _alter_time_value():
    time_val = select_fxtime.value
    if time_val == select_fxtime.end:
        play_buttons.active = 1
    else:
        select_fxtime.value = time_val + 1


@gen.coroutine
def _reset_time_value():
    select_fxtime.value = 0
    play_buttons.active = 1


def update_data(attr, old, new):
    try:
        doc.add_timeout_callback(_update_data, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_data(update_range=False):
    logging.debug('Updating data...')
    valid_date = times[int(select_fxtime.value)]
    masked_regrid, X, Y = load_data(valid_date)
    xn = X[0]
    yn = Y[:, 0]
    local_data_source.data.update({'masked_regrid': [masked_regrid],
                                   'xn': [xn], 'yn': [yn],
                                   'valid_date': [valid_date]})
    try:
        doc.add_next_tick_callback(partial(_update_map, update_range))
    except ValueError:
        pass
    try:
        doc.add_next_tick_callback(_update_histogram)
    except ValueError:
        pass
    try:
        doc.add_next_tick_callback(_move_hist_line)
    except ValueError:
        pass
    try:
        doc.add_next_tick_callback(_update_tseries_pt)
    except ValueError:
        pass

    logging.debug('Done updating data')


def update_models(attr, old, new):
    try:
        doc.add_timeout_callback(_update_models, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_models(update_range=False):
    logging.debug('Updating models...')
    date = file_dict[select_day.value]
    models = get_models(date)
    select_model.options = models

    thelabel = ''
    for m, disabled in models:
        if m == select_model.value and not disabled:
            thelabel = m
        if not disabled and not thelabel:
            thelabel = m
    select_model.value = thelabel
    doc.add_next_tick_callback(partial(_update_file, update_range))


def update_file(attr, old, new):
    try:
        doc.add_timeout_callback(_update_file, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_file(update_range=False):
    date = file_dict[select_day.value]
    load_file(select_model.value, date)
    select_fxtime.end = len(times) - 1
    if select_fxtime.value > select_fxtime.end:
        select_fxtime.value = select_fxtime.end

    try:
        doc.add_next_tick_callback(partial(_update_data, update_range))
    except ValueError:
        pass

    try:
        doc.add_next_tick_callback(_update_tseries)
    except ValueError:
        pass


def move_click_marker(event):
    try:
        doc.add_timeout_callback(partial(_move_click_marker, event), 50)
    except ValueError as e:
        pass


def time_setter(index):
    if isinstance(index, pd.DatetimeIndex):
        return (index.tz_convert('MST').tz_localize(None).values.astype(int) /
                10**6)
    elif isinstance(index, pd.Timestamp):
        return index.tz_convert('MST').tz_localize(None).value / 10**6
    else:
        raise AttributeError


@gen.coroutine
def _move_click_marker(event):
    x = event.x
    y = event.y

    xn = local_data_source.data['xn'][0]
    yn = local_data_source.data['yn'][0]

    x_idx = np.abs(xn - x).argmin()
    y_idx = np.abs(yn - y).argmin()
    lat, lon = webmerc_to_latlon(xn[x_idx], yn[y_idx])

    hover_pt.data.update({'x': np.array([xn[x_idx]]),
                          'y': np.array([yn[y_idx]]),
                          'x_idx': np.array([x_idx]),
                          'y_idx': np.array([y_idx]),
                          'lat': np.array([lat]),
                          'lon': np.array([lon])})
    doc.add_next_tick_callback(_move_hist_line)
    doc.add_next_tick_callback(_update_div_text)
    doc.add_next_tick_callback(_update_tseries)


@gen.coroutine
def _update_tseries():
    tseries_fig.title.text = (
        f'UA-HAS WRF {select_day.value} {select_model.value} at '
        f'({hover_pt.data["lat"][0]:0.2f}, {hover_pt.data["lon"][0]:0.2f})')
    x_idx = hover_pt.data['x_idx'][0]
    y_idx = hover_pt.data['y_idx'][0]

    line_data = load_tseries(x_idx, y_idx)
    if len(line_data.dropna().values) < 1:
        tseries_source.data.update({'values': np.array([config.MAX_VAL]),
                                    'time': np.array([0])})
    else:
        tseries_source.data.update({'values': line_data.values,
                                    'time': time_setter(times)})
    try:
        doc.add_next_tick_callback(_update_tseries_pt)
    except ValueError:
        pass


@gen.coroutine
def _update_tseries_pt():
    val = info_data.data['current_val'][0]
    if val == np.nan:
        val = 0
    curtime = local_data_source.data['valid_date'][0]
    curpt_source.data.update({'value': np.array([val]),
                              'time': np.array([time_setter(curtime)])})


@gen.coroutine
def _move_hist_line():
    x_idx = hover_pt.data['x_idx'][0]
    y_idx = hover_pt.data['y_idx'][0]
    masked_regrid = local_data_source.data['masked_regrid'][0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val = float(masked_regrid[y_idx, x_idx])
    info_data.data.update({'current_val': np.array([val])})
    try:
        doc.add_next_tick_callback(_update_div_text)
    except ValueError:
        pass

    if val <= config.MIN_VAL or val == np.nan:
        val = config.MIN_VAL * 1.05
    elif val > config.MAX_VAL:
        val = config.MAX_VAL * .99
    line_source.data.update({'x': np.array([val, val])})


def webmerc_to_latlon(x, y):
    R = 6378137.0
    lon = np.degrees(x / R)
    lat = np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)
    return lat, lon


@gen.coroutine
def _update_div_text():
    current_val = info_data.data['current_val'][0]
    mean = info_data.data['mean'][0]
    bin_width = info_data.data['bin_width'][0]
    lat = hover_pt.data['lat'][0]
    lon = hover_pt.data['lon'][0]
    info_div.text = info_text.format(current_val=current_val,
                                     mean=mean,
                                     bin_width=bin_width,
                                     lat=lat, lon=lon)


doc = curdoc()
# python callbacks
map_fig.x_range.on_change('start', update_histogram)
map_fig.x_range.on_change('end', update_histogram)
map_fig.y_range.on_change('start', update_histogram)
map_fig.y_range.on_change('end', update_histogram)
map_fig.on_event(events.Tap, move_click_marker)
map_fig.on_event(events.Press, move_click_marker)

select_day.on_change('value', update_models)
select_model.on_change('value', update_file)
select_fxtime.on_change('value', update_data)
play_buttons.on_change('active', animate_times)

for thing in (select_day, select_model, select_fxtime, play_buttons,
              map_fig, hist_fig, tseries_fig, info_div):
    doc.add_root(thing)
doc.add_next_tick_callback(partial(_update_models, True))
doc.add_timeout_callback(_update_data, 1000)
doc.title = config.TITLE
doc.template_variables.update({
    'menu_vars': config.MENU_VARS,
    'prefix': config.PREFIX,
    'ga_tracking_id': config.GA_TRACKING_ID})
try:
    custom_model_code = sys.argv[2]
except IndexError:
    pass
else:
    doc.template_variables['custom_model_code'] = custom_model_code
