from bokeh.models.callbacks import Callback
from bokeh.core.properties import (List, Either, String, Tuple, Bool,
                                   Instance)
from bokeh.models.widgets.inputs import InputWidget


class DisabledSelect(InputWidget):
    ''' Single-select widget.
    '''
    __js_implementation__ = 'disabled_select.ts'

    options = List(Either(String, Tuple(String, Bool)), help="""
    Available selection options. Options may be provided either as a list of
    possible string values, or as a list of tuples, each of the form
    ``(value, disabled)``.
    """)

    value = String(default="", help="""
    Initial or selected value.
    """)

    callback = Instance(Callback, help="""
    A callback to run in the browser whenever the current Select dropdown
    value changes.
    """)
