import * as p from "core/properties"

import {color2hex} from "core/util/color"
import {ColorMapper} from "models/mappers/color_mapper"

export class BinnedColorMapper extends ColorMapper
  type: "BinnedColorMapper"

  @define {
      alpha: [ p.Number, 1.0 ]
  }

  initialize: (attrs, options) ->
    super(attrs, options)
    @_nan_color = @_build_palette([color2hex(@nan_color)])[0]

  v_map_screen: (data, image_glyph=false) ->
    values = @_get_values(data, @_palette, image_glyph)
    buf = new ArrayBuffer(data.length * 4)
    hexalpha = Math.floor(@alpha * 255)
    if @_little_endian
      color = new Uint8Array(buf)
      for i in [0...data.length]
        value = values[i]
        ind = i*4
        # Bitwise math in JS is limited to 31-bits, to handle 32-bit value
        # this uses regular math to compute alpha instead (see issue #6755)
        color[ind] = Math.floor((value/4278190080.0) * 255)
        color[ind+1] = (value & 0xff0000) >> 16
        color[ind+2] = (value & 0xff00) >> 8
        color[ind+3] = value & hexalpha
    else
      color = new Uint32Array(buf)
      for i in [0...data.length]
        value = values[i]
        color[i] = (value << 8) | hexalpha
    return buf

  _get_values: (data, palette, image_glyph=false) ->
    values = []

    for d in data
      if isNaN(d)
        values.push(nan_color)
        continue
      i = Math.floor(d)
      values.push(palette[i])
    return values
