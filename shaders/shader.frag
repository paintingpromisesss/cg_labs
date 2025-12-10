#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_shadow_coord;

layout (location = 0) out vec4 final_color;

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

layout (binding = 2) uniform sampler2DShadow shadow_map;
layout (binding = 3) uniform sampler2D main_texture;

void main() {
	vec3 N = normalize(f_normal);
	vec3 L = normalize(-light_direction.xyz);
	vec3 V = normalize(camera_position.xyz - f_position);
	vec3 H = normalize(L + V);

	vec3 lightColor = light_color_intensity.rgb * light_color_intensity.w;
	vec3 ambient = ambient_color.rgb * albedo_color;

	// Diffuse
	float diff = max(dot(N, L), 0.0);
	vec3 diffuse = diff * albedo_color * lightColor;

	// Specular
	float spec = 0.0;
	if (diff > 0.0) {
		spec = pow(max(dot(N, H), 0.0), 32.0);
	}
	vec3 specular = lightColor * spec;

	// Shadow
	float shadow = 1.0;
	if (f_shadow_coord.z > -1.0 && f_shadow_coord.z < 1.0) {
		vec3 projCoords = f_shadow_coord.xyz / f_shadow_coord.w;
		
		// PCF
		float shadow_sum = 0.0;
		vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
		for(int x = -1; x <= 1; ++x)
		{
			for(int y = -1; y <= 1; ++y)
			{
				float pcfDepth = texture(shadow_map, projCoords + vec3(x, y, 0.0) * vec3(texelSize, 0.0)); 
				shadow_sum += pcfDepth;
			}    
		}
		shadow = shadow_sum / 9.0;
	}

	vec4 tex_color = texture(main_texture, f_uv);
	
	vec3 lighting = ambient + (diffuse + specular) * shadow;
	final_color = vec4(lighting, 1.0) * tex_color;
}
