// This is just a modified version of Bokeh's SelectView found in
// /src/lib/models/widgets/selectbox.ts of the src repository as of
// version 3.3.2
import {select, option, optgroup, empty, append} from "core/dom"
import {isString, isArray} from "core/util/types"
import {entries} from "core/util/object"
import type * as p from "core/properties"

import {InputWidget, InputWidgetView} from "models/widgets/input_widget"
import * as inputs from "styles/widgets/inputs.css"

export class DisabledSelectView extends InputWidgetView {
  declare model: DisabledSelect

  declare input_el: HTMLSelectElement

  override connect_signals(): void {
    super.connect_signals()
    const {value, options} = this.model.properties
    this.on_change(value, () => {
      this._update_value()
    })
    this.on_change(options, () => {
      empty(this.input_el)
      append(this.input_el, ...this.options_el())
      this._update_value()
    })
  }

  private _known_values = new Set<string>()

  protected options_el(): HTMLOptionElement[] | HTMLOptGroupElement[] {
    const {_known_values} = this
    _known_values.clear()

    function build_options(values: (string | [string, boolean])[]): HTMLOptionElement[] {
      return values.map((el) => {
        let value
	let disabled
        if (isString(el)) {
          value = el
          disabled = false
	}  else
          [value, disabled] = el

        _known_values.add(value)
        return option({value, disabled}, value)
      })
    }

    const {options} = this.model
    if (isArray(options))
      return build_options(options)
    else
      return entries(options).map(([label, values]) => optgroup({label}, build_options(values)))
  }

  override render(): void {
    super.render()

    this.input_el = select({
      class: inputs.input,
      name: this.model.name,
      disabled: this.model.disabled,
    }, this.options_el())

    this._update_value()

    this.input_el.addEventListener("change", () => this.change_input())
    this.group_el.appendChild(this.input_el)
  }

  override change_input(): void {
    const value = this.input_el.value
    this.model.value = value
    super.change_input()
  }

  protected _update_value(): void {
    const {value} = this.model
    if (this._known_values.has(value))
      this.input_el.value = value
    else
      this.input_el.removeAttribute("value")
  }
}

export namespace DisabledSelect {
  export type Attrs = p.AttrsOf<Props>

  export type Props = InputWidget.Props & {
    value: p.Property<string>
    options: p.Property<(string | [string, boolean])[] | {[key: string]: (string | [string, boolean])[]}>
  }
}

export interface DisabledSelect extends DisabledSelect.Attrs {}

export class DisabledSelect extends InputWidget {
  declare properties: DisabledSelect.Props
  declare __view_type__: DisabledSelectView

  constructor(attrs?: Partial<DisabledSelect.Attrs>) {
    super(attrs)
  }

  static {
    this.prototype.default_view = DisabledSelectView

    this.define<DisabledSelect.Props>(({String, Boolean, Array, Tuple, Dict, Or}) => {
      const Options = Array(Or(String, Tuple(String, Boolean)))
      return {
        value:   [ String, "" ],
        options: [ Or(Options, Dict(Options)), [] ],
      }
    })
  }
}
