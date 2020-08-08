#version 330 core

const vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);

layout(location = 0) in float center;
layout(location = 1) in float radius;
layout(location = 2) in vec4 color_vert;

out vec4 color_frag;

uniform float camera_offset;
uniform float camera_scale;

void main() {
    color_frag = color_vert;

    float pos = center + radius * positions[gl_VertexID].x;
    pos = camera_offset + camera_scale * pos;
    gl_Position = vec4(pos, positions[gl_VertexID].y, 0.0, 1.0);
}
