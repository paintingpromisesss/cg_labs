#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec4 light_direction;
	vec4 light_color_intensity;
	vec4 camera_position;
	vec4 ambient_color;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
};

void main() {
	gl_Position = light_view_projection * model * vec4(v_position, 1.0);
}
