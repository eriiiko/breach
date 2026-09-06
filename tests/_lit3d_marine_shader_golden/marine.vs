#version 330
// Lit-marine vertex shader. The mesh is CPU-skinned upstream
// (update_model_animation), so vertexPosition/vertexNormal are already POSED —
// no bone skinning here (declaring boneMatrices would double-transform). We
// only lift the posed vertex to world space for the fragment's field lookup.
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;

uniform mat4 mvp;
uniform mat4 matModel;    // auto-uploaded by DrawMesh (loc auto-populated)
uniform mat4 matNormal;   // = transpose(inverse(matModel)), world-space normals

out vec2 fragTexCoord;
out vec3 fragWorldPos;
out vec3 fragWorldNormal;

void main() {
    fragTexCoord = vertexTexCoord;
    fragWorldPos = (matModel * vec4(vertexPosition, 1.0)).xyz;
    fragWorldNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
