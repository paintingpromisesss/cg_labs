#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (set = 0, binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
};

// Текстура материала (set = 1, binding = 0)
layout (set = 1, binding = 0) uniform sampler2D material_texture;

void main() {
	// Сэмплируем текстуру по UV координатам
	vec4 tex_color = texture(material_texture, f_uv);
	
	// Умножаем цвет текстуры на albedo цвет материала
	final_color = tex_color * vec4(albedo_color, 1.0f);
}
