#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    float x = float(((uint(gl_VertexIndex) + 2) / 3) % 2);
    float y = float(((uint(gl_VertexIndex) + 1) / 3) % 2);
    gl_Position = vec4(-1.0 + x * 2.0, -1.0 + y * 2.0, 0.0, 1.0);
}
