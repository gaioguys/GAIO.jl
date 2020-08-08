#version 330 core

const vec3 positions[14] = vec3[](
    vec3(-1.f, 1.f, 1.f),
    vec3(1.f, 1.f, 1.f),
    vec3(-1.f, -1.f, 1.f),
    vec3(1.f, -1.f, 1.f),
    vec3(1.f, -1.f, -1.f),
    vec3(1.f, 1.f, 1.f),
    vec3(1.f, 1.f, -1.f),
    vec3(-1.f, 1.f, 1.f),
    vec3(-1.f, 1.f, -1.f),
    vec3(-1.f, -1.f, 1.f),
    vec3(-1.f, -1.f, -1.f),
    vec3(1.f, -1.f, -1.f),
    vec3(-1.f, 1.f, -1.f),
    vec3(1.f, 1.f, -1.f)
);

layout(location = 0) in vec3 center;
layout(location = 1) in vec3 radius;
layout(location = 2) in vec4 color_vert;

uniform mat4 camera;

out vec4 color_frag;

void main() {
    vec3 color = 0.5f * (positions[gl_VertexID] + vec3(1.f, 1.f, 1.f));
    color = mix(color_vert.xyz, color, 0.3);
    color_frag = vec4(color.x, color.y, color.z, color_vert.w);

    vec3 pos = center + radius * positions[gl_VertexID];
    gl_Position = camera * vec4(pos.x, pos.y, pos.z, 1.0);
}
