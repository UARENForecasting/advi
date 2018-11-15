import * as p from "core/properties"
import {Arrayable} from "core/types"
import {ColorMapper} from "models/mappers/color_mapper"

function _component2hex(v: number | string): string {
    const h = Number(v).toString(16)
    return h.length == 1 ? `0${h}` : h
}

export namespace BinnedColorMapper {
    export interface Attrs extends ColorMapper.Attrs {
        alpha: number
    }

    export interface Props extends ColorMapper.Props {}
}

export interface BinnedColorMapper extends BinnedColorMapper.Attrs {}

export class BinnedColorMapper extends ColorMapper {

    properties: BinnedColorMapper.Props

    constructor(attrs?: Partial<BinnedColorMapper.Attrs>) {
        super(attrs)
    }

    static initClass(): void {
        this.prototype.type = "BinnedColorMapper"
    }

    protected _v_compute<T>(data: Arrayable<number>, values: Arrayable<T>,
                            palette: Arrayable<t>, colors: {nan_color: T}): void {
        const {nan_color} = colors
        const max_key = palette.length - 1

        for (let i = 0, end = data.length; i < end; i++) {
            const d = data[i]

            if (isNaN(d)) {
                values[i] = nan_color
                continue
            }

            const key = Math.floor(d)
            if (key < 0)
                values[i] = palette[0]
            else if (key > max_key)
                values[i] = palette[max_key]
            else
                values[i] = palette[key]
        }
    }
}
BinnedColorMapper.initClass()
