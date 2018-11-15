from bokeh.models.mappers import ColorMapper


class BinnedColorMapper(ColorMapper):
    """
    Map integers to the palette bin.
    """
    __js_implementation__ = 'binned_color_mapper.ts'
