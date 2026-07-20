"""Introspect the Quaternius marine model + custom-shader loc behaviour.

Confirms, empirically in THIS raylib wheel:
  - meshCount / materialCount / meshMaterial mapping (materials 1 & 2 live?)
  - whether a custom material shader declaring `matModel`/`matNormal` gets
    those loc slots auto-populated (SHADER_LOC_MATRIX_MODEL / _NORMAL != -1)
  - the MATERIAL_MAP_* -> sampler-name (texture0/1/2) expectation
Run: conda run -n data python scratchpad/introspect_model.py
"""
from pathlib import Path
import pyray as rl

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "models" / "marine" / "AnimationLibrary_Godot_Standard.gltf"

rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
rl.init_window(320, 240, "introspect")

model = rl.load_model(str(MODEL))
print("meshCount   =", model.meshCount)
print("materialCount =", model.materialCount)
for i in range(model.meshCount):
    print(f"  mesh[{i}] boneCount={model.meshes[i].boneCount} "
          f"meshMaterial={model.meshMaterial[i]} "
          f"vertexCount={model.meshes[i].vertexCount}")
for m in range(model.materialCount):
    mat = model.materials[m]
    alb = mat.maps[rl.MaterialMapIndex.MATERIAL_MAP_ALBEDO].texture
    print(f"  material[{m}] shader.id={mat.shader.id} albedo_tex.id={alb.id}")

# --- Try loading a minimal custom shader from memory that declares matModel/matNormal
VS = """#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
out vec2 fragTexCoord;
out vec3 worldPos;
out vec3 worldNormal;
void main() {
    fragTexCoord = vertexTexCoord;
    worldPos = (matModel * vec4(vertexPosition, 1.0)).xyz;
    worldNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""
FS = """#version 330
in vec2 fragTexCoord;
in vec3 worldPos;
in vec3 worldNormal;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform vec4 colDiffuse;
out vec4 finalColor;
void main() {
    vec3 a = texture(texture0, fragTexCoord).rgb;
    vec3 la = texture(texture1, fragTexCoord).rgb;
    vec3 lb = texture(texture2, fragTexCoord).rgb;
    finalColor = vec4(a * colDiffuse.rgb + 0.0*(la+lb) + 0.0*worldPos + 0.0*worldNormal, 1.0);
}
"""
sh = rl.load_shader_from_memory(VS, FS)
print("custom shader id =", sh.id)
print("loc matModel  =", rl.get_shader_location(sh, "matModel"),
      " SHADER_LOC_MATRIX_MODEL slot =",
      sh.locs[rl.ShaderLocationIndex.SHADER_LOC_MATRIX_MODEL])
print("loc matNormal =", rl.get_shader_location(sh, "matNormal"),
      " SHADER_LOC_MATRIX_NORMAL slot =",
      sh.locs[rl.ShaderLocationIndex.SHADER_LOC_MATRIX_NORMAL])
print("loc mvp       =", rl.get_shader_location(sh, "mvp"),
      " SHADER_LOC_MATRIX_MVP slot =",
      sh.locs[rl.ShaderLocationIndex.SHADER_LOC_MATRIX_MVP])
print("loc texture0  =", rl.get_shader_location(sh, "texture0"),
      " MAP_ALBEDO slot =", sh.locs[rl.ShaderLocationIndex.SHADER_LOC_MAP_ALBEDO])
print("loc texture1  =", rl.get_shader_location(sh, "texture1"),
      " MAP_METALNESS slot =", sh.locs[rl.ShaderLocationIndex.SHADER_LOC_MAP_METALNESS])
print("loc texture2  =", rl.get_shader_location(sh, "texture2"),
      " MAP_NORMAL slot =", sh.locs[rl.ShaderLocationIndex.SHADER_LOC_MAP_NORMAL])
print("loc colDiffuse=", rl.get_shader_location(sh, "colDiffuse"),
      " COLOR_DIFFUSE slot =", sh.locs[rl.ShaderLocationIndex.SHADER_LOC_COLOR_DIFFUSE])

rl.unload_shader(sh)
rl.unload_model(model)
rl.close_window()
print("OK")
