from bokeh.core.properties import Percent
from bokeh.models.mappers import ColorMapper


class BinnedColorMapper(ColorMapper):
    """
    Map integers to the palette bin.
    """
    __js_implementation__ = 'binned_color_mapper.coffee'

    alpha = Percent(default=1.0, help="""
    The alpha (0.0 to 1.0) to apply to all colors.
    """)
