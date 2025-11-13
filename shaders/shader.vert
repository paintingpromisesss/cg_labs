#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

// Выходные данные для fragment shader
layout (location = 0) out vec3 f_position;  // Позиция в world space
layout (location = 1) out vec3 f_normal;    // Нормаль в world space
layout (location = 2) out vec2 f_uv;

// Uniform buffer: камера и проекция
layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
};

// Dynamic uniform buffer: трансформация и материал модели
layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float _pad0;
	vec3 specular_color;
	float shininess;
};

void main() {
	// Трансформируем позицию в world space
	vec4 world_position = model * vec4(v_position, 1.0f);
	
	// Normal matrix: transpose(inverse(model)) - правильная трансформация нормалей
	// Нужна для корректной работы при неравномерном масштабировании
	mat3 normal_matrix = transpose(inverse(mat3(model)));
	vec3 world_normal = normal_matrix * v_normal;

	// Финальная позиция в clip space
	gl_Position = view_projection * world_position;

	// Передаём данные в fragment shader
	f_position = world_position.xyz;
	f_normal = normalize(world_normal);
	f_uv = v_uv;
}
