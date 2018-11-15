import {empty, label, select, option, optgroup} from "core/dom"
import {isString, isArray} from "core/util/types"
import {logger} from "core/logging"
import * as p from "core/properties"

import {InputWidget, InputWidgetView} from "models/widgets/input_widget"

export class DisabledSelectView extends InputWidgetView {
    model: DisabledSelect

    protected selectEl: HTMLSelectElement

    initialize(options: any): void {
        super.initialize(options)
        this.render()
    }

    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.change, () => this.render())
    }

    build_options(values: (string | [string, bool])[]): HTMLElement[] {
        return values.map((el) => {
            let value, disabled
            if (isString(el)) {
                value = el
                disabled = false
            } else {
                [value, disabled] = el
            }

            const selected = this.model.value == value
            return option({selected: selected, value: value,
                           disabled: disabled}, value)
        })
    }

    render(): void {
        super.render()
        empty(this.el)

        const labelEl = label({for: this.model.id}, this.model.title)
        this.el.appendChild(labelEl)

        let contents: HTMLElement[]
        if (isArray(this.model.options))
            contents = this.build_options(this.model.options)
        else {
            contents = []
            const options = this.model.options
            for (const key in options) {
                const value = options[key]
                contents.push(optgroup({label: key}, this.build_options(value)))
            }
        }

        this.selectEl = select({
            class: "bk-widget-form-input",
            id: this.model.id,
            name: this.model.name,
            disabled: this.model.disabled}, contents)

        this.selectEl.addEventListener("change", () => this.change_input())
        this.el.appendChild(this.selectEl)
    }

    change_input(): void {
        const value = this.selectEl.value
        logger.debug(`selectbox: value = ${value}`)
        this.model.value = value
        super.change_input()
    }
}

export namespace DisabledSelect {
    export interface Attrs extends InputWidget.Attrs {
        value: string
        options: (string | [string, string])[] | {[key: string]: (string | [string, string])[]}
    }

    export interface Props extends InputWidget.Props {}
}

export interface DisabledSelect extends DisabledSelect.Attrs {}

export class DisabledSelect extends InputWidget {

    properties: DisabledSelect.Props

    constructor(attrs?: Partial<DisabledSelect.Attrs>) {
        super(attrs)
    }

    static initClass(): void {
        this.prototype.type = "DisabledSelect"
        this.prototype.default_view = DisabledSelectView

        this.define({
            value:   [ p.String, '' ],
            options: [ p.Any,    [] ], // TODO (bev) is this used?
        })
    }
}

DisabledSelect.initClass()
