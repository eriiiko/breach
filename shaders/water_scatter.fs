#version 330
// water_scatter.fs -- the scatter caustic (#97 W4): one scattered light
// triangle's density, added (blend ONE, ONE) into the light image: R = the
// light (incident 1), G = how many triangles covered this cell (diagnostic).
flat in float v_ratio;
out vec4 finalColor;
void main() {
    finalColor = vec4(v_ratio, 1.0, 0.0, 0.0);
}
