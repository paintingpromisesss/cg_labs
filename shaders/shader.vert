#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_shadow_coord;

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

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 
);

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = normal.xyz;
	f_uv = v_uv;
	
	f_shadow_coord = biasMat * light_view_projection * position;
}
