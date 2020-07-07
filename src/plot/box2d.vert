#version 330 core
#extension GL_ARB_explicit_uniform_location : require

const vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);

layout(location = 0) in vec2 center;
layout(location = 1) in vec2 radius;

layout(location = 0) uniform vec2 camera_offset;
layout(location = 1) uniform vec2 camera_scale;

void main() {
    vec2 pos = center + radius * positions[gl_VertexID];

    pos = camera_offset + camera_scale * pos;

    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
}
