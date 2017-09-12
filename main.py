from collections import OrderedDict
import datetime as dt
from functools import partial
import logging
from pathlib import Path
import os


from bokeh import events
from bokeh.colors import RGB
from bokeh.layouts import gridplot, column, row
from bokeh.models import (
    Range1d, LinearColorMapper, ColorBar, FixedTicker,
    ColumnDataSource, CustomJS, WMTSTileSource, Spacer,
    Slider)
from bokeh.models.widgets import Select, Div
from bokeh.plotting import figure, curdoc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable, get_cmap
import numpy as np
import pandas as pd
import tables
from tornado import gen


from models.disabled_select import DisabledSelect


ALPHA = 0.7
DATA_DIRECTORY = os.getenv('ARTSY_WRF_DATADIR', '~/.wrf')
POSSIBLE_MODELS = ('WRFGFS_00Z', 'WRFGFS_06Z', 'WRFGFS_12Z',
                   'WRFNAM_00Z', 'WRFNAM_06Z', 'WRFNAM_12Z')


def k_to_f(temp):
    c = temp - 273.15
    f = c * 9 / 5 + 32
    return f


curdir = os.path.basename(os.path.dirname(__file__))
if curdir == 'ghi':
    MIN_VAL = 0
    MAX_VAL = 1200
    VAR = 'SWDNB'
    CMAP = 'viridis'
    CONVFUNC = lambda x: x
    XLABEL = 'GHI (W/m^2)'
else:
    MIN_VAL = 0
    MAX_VAL = 120
    VAR = 'T2'
    CMAP = 'plasma'
    CONVFUNC = k_to_f
    XLABEL = 'Temperature (°F)'

def load_file(model, fx_date='latest'):
    dir = os.path.expanduser(DATA_DIRECTORY)
    if fx_date == 'latest':
        p = Path(dir)
        model_dir = sorted([pp for pp in p.rglob(f'*{model}')],
                           reverse=True)[0]
    else:
        model_dir = os.path.join(dir, fx_date.strftime('%Y/%m/%d'),
                                 strpmodel(model))

    path = os.path.join(model_dir,
                        f'{VAR}.h5')

    global h5file
    try:
        h5file.close()
    except:
        pass
    h5file = tables.open_file(path, mode='r')
    global times
    times = pd.DatetimeIndex(
        h5file.get_node('/times')[:]).tz_localize('UTC')


def load_data(valid_date):
    strformat = '%Y%m%dT%H%MZ'
    data = h5file.get_node(f'/{valid_date.strftime(strformat)}')[:]
    regridded_data = CONVFUNC(data)
    regridded_data[np.isnan(regridded_data)] = -999

    X = h5file.get_node('/X')[:]
    Y = h5file.get_node('/Y')[:]
    masked_regrid = np.ma.masked_less(regridded_data, MIN_VAL)
    return masked_regrid, X, Y


def find_fx_times():
    p = Path(DATA_DIRECTORY).expanduser()
    out = OrderedDict()
    for pp in sorted(p.rglob(f'*WRF*')):
        try:
            datetime = dt.datetime.strptime(''.join(pp.parts[-4:-1]),
                                            '%Y%m%d')
        except ValueError:
            logging.debug('%s does not conform to expected format', pp)
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
    dir = os.path.join(DATA_DIRECTORY, date.strftime('%Y/%m/%d'))
    p = Path(dir).expanduser()
    disabled = {model: True for model in POSSIBLE_MODELS}
    for pp in p.iterdir():
        m = pp.parts[-1]
        disabled[m] = False
    mld = [(strfmodel(k), v) for k, v in disabled.items()]
    return mld


# setup the coloring
levels = MaxNLocator(nbins=25).tick_values(MIN_VAL, MAX_VAL)
cmap = get_cmap(CMAP)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
sm = ScalarMappable(norm=norm, cmap=cmap)
color_pal = [RGB(*val).to_hex() for val in
             sm.to_rgba(levels, bytes=True, norm=True)[:-1]]
color_mapper = LinearColorMapper(color_pal, low=sm.get_clim()[0],
                                 high=sm.get_clim()[1])
ticker = FixedTicker(ticks=levels[::3])
cb = ColorBar(color_mapper=color_mapper, location=(0, 0),
              scale_alpha=ALPHA, ticker=ticker)

# make the bokeh figures without the data yet
width = 600
height = 400
sfmt = '%Y-%m-%d %HZ'
tools = 'pan, box_zoom, reset, save'
map_fig = figure(plot_width=width, plot_height=height,
                 y_axis_type=None, x_axis_type=None,
                 toolbar_location='left', tools=tools + ', wheel_zoom',
                 active_scroll='wheel_zoom',
                 title='')

rgba_img_source = ColumnDataSource(data={'image': [], 'x': [], 'y': [],
                                         'dw': [], 'dh': []})
rgba_img = map_fig.image_rgba(image='image', x='x', y='y', dw='dw', dh='dh',
                              source=rgba_img_source)


# Need to use this and not bokeh.tile_providers.STAMEN_TONER
# https://github.com/bokeh/bokeh/issues/4770
STAMEN_TONER = WMTSTileSource(
    url='https://stamen-tiles.a.ssl.fastly.net/toner-lite/{Z}/{X}/{Y}.png',
    attribution=(
        'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
        'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0'
        '</a>. Map data by <a href="http://openstreetmap.org">OpenStreetMap'
        '</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
    )
)
map_fig.add_tile(STAMEN_TONER)
map_fig.add_layout(cb, 'right')

# Make the histogram figure
hist_fig = figure(plot_width=height, plot_height=height,
                  toolbar_location='right',
                  x_axis_label=XLABEL,
                  y_axis_label='Counts', tools=tools + ', ywheel_zoom',
                  active_scroll='ywheel_zoom',
                  x_range=Range1d(start=MIN_VAL, end=MAX_VAL))

# make histograms
bin_width = levels[1] - levels[0]
bin_centers = levels[:-1] + bin_width / 2
hist_sources = [ColumnDataSource(data={'x': [bin_centers[i]],
                                       'top': [3.0e6],
                                       'color': [color_pal[i]],
                                       'bottom': [0],
                                       'width': [bin_width]})
                for i in range(len(bin_centers))]
for source in hist_sources:
    hist_fig.vbar(x='x', top='top', width='width', bottom='bottom',
                  color='color', fill_alpha=ALPHA, source=source)

# line and point on map showing tapped location value
line_source = ColumnDataSource(data={'x': [-1, -1], 'y': [0, 1]})
hist_fig.line(x='x', y='y', color='red', source=line_source, alpha=ALPHA)
hover_pt = ColumnDataSource(data={'x': [0], 'y': [0], 'x_idx': [0],
                                  'y_idx': [0]})
map_fig.x(x='x', y='y', size=10, color='red', alpha=ALPHA,
          source=hover_pt, level='overlay')

file_dict = find_fx_times()
dates = list(file_dict.keys())[::-1]
select_day = Select(title='Initialization Day', value=dates[0], options=dates)
select_model = DisabledSelect(title='Initialization', value='',
                              options=[])
times = []
select_fxtime = Slider(title='Forecast Hour', start=0, end=1, value=0,
                       name='timeslider')
info_data = ColumnDataSource(data={'current_val': [0], 'mean': [0]})
info_text = """
<div class="well">
<b>Selected Value:</b> {current_val:0.1f} <b>Mean:</b> {mean:0.1f}
</div>
"""
info_div = Div(sizing_mode='scale_width')

# Setup the updates for all the data
local_data_source = ColumnDataSource(data={'masked_regrid': [0], 'xn': [0],
                                           'yn': [0],
                                           'valid_date': [dt.datetime.now()]})


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
        new_subset.clip(max=MAX_VAL), bins=levels,
        range=(levels.min(), levels.max()))
    line_source.data.update({'y': [0, counts.max()]})
    for i, source in enumerate(hist_sources):
        source.data.update({'top': [counts[i]]})
    logging.debug('Done updating histogram')

    info_data.data.update({'mean': [float(new_subset.mean())]})
    doc.add_next_tick_callback(_update_div_text)


def update_map(attr, old, new):
    try:
        doc.add_timeout_callback(_update_histogram, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_map(update_range=False):
    logging.debug('Updating map...')
    valid_date = local_data_source.data['valid_date'][0]
    model = select_model.value
    title = f'WRF {XLABEL} valid at {valid_date.strftime(sfmt)}'
    map_fig.title.text = title
    masked_regrid = local_data_source.data['masked_regrid'][0]
    xn = local_data_source.data['xn'][0]
    yn = local_data_source.data['yn'][0]
    rgba_vals = sm.to_rgba(masked_regrid, bytes=True, alpha=ALPHA)
    dx = xn[1] - xn[0]
    dy = yn[1] - yn[0]
    rgba_img_source.data.update({'image': [rgba_vals],
                                 'x': [xn[0] - dx / 2],
                                 'y': [yn[0] - dy / 2],
                                 'dw': [xn[-1] - xn[0] + dx],
                                 'dh': [yn[-1] - yn[0] + dy]})
    if update_range:
        map_fig.x_range.start = xn[0]
        map_fig.x_range.end = xn[-1]
        map_fig.y_range.start = yn[0]
        map_fig.y_range.end = yn[-1]
    logging.debug('Done updating map')


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
    curdoc().add_next_tick_callback(partial(_update_map, update_range))
    curdoc().add_timeout_callback(_update_histogram, 10)
    curdoc().add_next_tick_callback(_move_hist_line)
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
    curdoc().add_next_tick_callback(partial(_update_file, update_range))


def update_file(attr, old, new):
    try:
        doc.add_timeout_callback(_update_file, 100)
    except ValueError:
        pass


@gen.coroutine
def _update_file(update_range=False):
    date = file_dict[select_day.value]
    load_file(select_model.value, date)
    options = [t.strftime('%Y-%m-%d %H:%MZ') for t in times]
    select_fxtime.end = len(options) - 1
    if select_fxtime.value > select_fxtime.end:
        select_fxtime.value = select_fxtime.end
    try:
        doc.add_next_tick_callback(partial(_update_data, update_range))
    except ValueError:
        pass


def move_click_marker(event):
    try:
        doc.add_timeout_callback(partial(_move_click_marker, event), 50)
    except ValueError:
        pass


@gen.coroutine
def _move_click_marker(event):
    x = event.x
    y = event.y

    xn = local_data_source.data['xn'][0]
    yn = local_data_source.data['yn'][0]

    x_idx = np.abs(xn - x).argmin()
    y_idx = np.abs(yn - y).argmin()

    hover_pt.data.update({'x': [xn[x_idx]], 'y': [yn[y_idx]],
                          'x_idx': [x_idx], 'y_idx': [y_idx]})
    curdoc().add_next_tick_callback(_move_hist_line)


@gen.coroutine
def _move_hist_line():
    x_idx = hover_pt.data['x_idx'][0]
    y_idx = hover_pt.data['y_idx'][0]
    masked_regrid = local_data_source.data['masked_regrid'][0]
    val = masked_regrid[y_idx, x_idx]
    info_data.data.update({'current_val': [float(val)]})
    doc.add_next_tick_callback(_update_div_text)

    if val <= MIN_VAL or val == np.nan:
        val = MIN_VAL * 1.05
    elif val > MAX_VAL:
        val = MAX_VAL * .99
    line_source.data.update({'x': [val, val]})


@gen.coroutine
def _update_div_text():
    current_val = info_data.data['current_val'][0]
    mean = info_data.data['mean'][0]
    info_div.text = info_text.format(current_val=current_val,
                                     mean=mean)


# python callbacks
map_fig.x_range.on_change('start', update_histogram)
map_fig.x_range.on_change('end', update_histogram)
map_fig.y_range.on_change('start', update_histogram)
map_fig.y_range.on_change('end', update_histogram)
map_fig.on_event(events.Tap, move_click_marker)

select_day.on_change('value', update_models)
select_model.on_change('value', update_file)
select_fxtime.on_change('value', update_data)

# layout the document
lay = column(row([select_day, select_model, select_fxtime, info_div]),
             row([map_fig, hist_fig]))
doc = curdoc()
doc.add_root(lay)
doc.add_next_tick_callback(partial(_update_models, True))
#doc.add_next_tick_callback(partial(_update_data, True))
