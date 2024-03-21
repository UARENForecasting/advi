# Includes modified Select class found in version 3.3.2 in src file
# /src/bokeh/models/widgets/input.py
from bokeh.models.callbacks import Callback
from bokeh.core.properties import (
	List,
	Either,
    String,
    Tuple,
    Bool,
	Dict,
    Instance,
	Null
)
from bokeh.models.widgets.inputs import InputWidget
from bokeh.util.compiler import TypeScript

class DisabledSelect(InputWidget):
    ''' Single-select widget with capability to have disabled options.

    '''
    __implementation__ = 'disabled_select.ts'
    # explicit __init__ to support Init signatures
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    options = List(Either(String, Tuple(String, Bool)), help="""
    Available selection options. Options may be provided either as a list of
    possible string values, or as a list of tuples, each of the form
    ``(value, disabled)``. In the latter case, the visible widget text for each
    value will be the value. Option groupings can be provided
    by supplying a dictionary object whose values are in the aforementioned
    list format
    """).accepts(List(Either(Null, String)), lambda v: [ "" if item is None else item for item in v ])

    value = String(default="", help="""
    Initial or selected value.
    """).accepts(Null, lambda _: "")
