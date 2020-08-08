#version 330 core

const vec2 positions[4] = vec2[](
    vec2(-1.0, 0.0),
    vec2( 1.0, 0.0),
    vec2( 1.0, 1.0),
    vec2(-1.0, 1.0)
);

layout(location = 0) in float center;
layout(location = 1) in float radius;
layout(location = 2) in float height;

uniform float camera_offset;
uniform float camera_scale;

void main() {
    float pos_x = center + radius * positions[gl_VertexID].x;
    pos_x = camera_offset + camera_scale * pos_x;

    float pos_y = 2.f * positions[gl_VertexID].y * height - 1.f;

    gl_Position = vec4(pos_x, pos_y, 0.0, 1.0);
}
