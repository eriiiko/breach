// water_scatter_slope.glsl -- PROTOTYPE (#97 reassessment, branch
// 97-reassess): the SLOPE PASS of the scatter caustic. Appended to
// shaders/water.fs with that file's main renamed
// (renderer/water.py::load_scatter_slope_fs), so the fine surface it writes
// is THE one the water draws (detailSurface, its flow map, its waves) --
// never a copy of the wave law. A direct-GL pass (pass_quad.vs's rects,
// the PingPong convention: GL row j = data row j, texels by gl_FragCoord).
// Each texel of the light image is one surface sample, at (cell +
// u_sgrid.w) cells from the image's origin (the twin's SCATTER_OFFSET), and
// writes (dh/dx, dh/dy, the per-pixel depth m, 1): the input of
// shaders/water_scatter.vs, which moves each sample by the refraction map.

uniform vec4 u_sgrid;   // (origin x, y in tiles, cells per tile, sample offset in cells)

void main() {
    vec2 cell = floor(gl_FragCoord.xy);
    vec2 tile = u_sgrid.xy + (cell + u_sgrid.w) / u_sgrid.z;
    vec2 world_uv = tile * u_texel;
    float tileDepth = texture(u_water, world_uv).b;
    if (tileDepth <= 0.0) {
        finalColor = vec4(0.0);
        return;
    }
    float relief_term = (u_has_height == 1 && u_height_scale > 0.0)
        ? (texture(u_height, u_art_uv_rect.xy + world_uv * u_art_uv_rect.zw).r
           - u_height_floor) : 0.0;
    float depthPos = max(tileDepth - relief_term * u_height_scale, 0.0);
    vec4 wtex = texture(u_water, world_uv);
    FlowLayers fl = flowLayers(world_uv, tileDepth);
    vec3 hess;
    vec3 detail = detailSurface(fl, wtex.a, false, hess);
    finalColor = vec4(detail.xy, depthPos, 1.0);
}
