import {empty, label, select, option} from "core/dom"
import {isString} from "core/util/types"
import {logger} from "core/logging"
import * as p from "core/properties"

import {InputWidget, InputWidgetView} from "models/widgets/input_widget"

export class DisabledSelectView extends InputWidgetView
  initialize: (options) ->
    super(options)
    @render()

  connect_signals: () ->
    super()
    @connect(@model.change, () -> @render())

  render: () ->
    super()
    empty(@el)

    labelEl = label({for: @model.id}, @model.title)
    @el.appendChild(labelEl)

    options = @model.options.map (opt) =>
      if isString(opt)
        value = opt
        disabled = false
      else
        [value, disabled] = opt

      selected = @model.value == value
      return option({selected: selected, value: value, disabled: disabled}, value)

    @selectEl = select({class: "bk-widget-form-input", id: @model.id, name: @model.name}, options)
    @selectEl.addEventListener("change", () => @change_input())
    @el.appendChild(@selectEl)

    return @

  change_input: () ->
    value = @selectEl.value
    logger.debug("selectbox: value = #{value}")
    @model.value = value
    super()

export class DisabledSelect extends InputWidget
  type: "DisabledSelect"
  default_view: DisabledSelectView

  @define {
    value:   [ p.String, '' ]
    options: [ p.Any,    [] ] # TODO (bev) is this used?
  }
