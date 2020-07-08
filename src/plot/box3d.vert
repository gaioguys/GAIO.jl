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

uniform mat4 camera;

out vec3 color;

void main() {
    vec3 pos = center + radius * positions[gl_VertexID];

    color = 0.5f * (positions[gl_VertexID] + vec3(1.f, 1.f, 1.f));

    gl_Position = camera * vec4(pos.x, pos.y, pos.z, 1.0);
}
