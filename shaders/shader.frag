#version 450

// Входные данные из vertex shader
layout (location = 0) in vec3 f_position;  // Позиция фрагмента в world space
layout (location = 1) in vec3 f_normal;    // Нормаль в world space
layout (location = 2) in vec2 f_uv;        // UV координаты

layout (location = 0) out vec4 final_color;

// Uniform buffer: камера
layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;  // Позиция камеры для вычисления specular
};

// Dynamic uniform buffer: материал модели
layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;      // Диффузный цвет (основной цвет поверхности)
	float _pad0;
	vec3 specular_color;    // Цвет бликов
	float shininess;        // Резкость бликов (2-256)
};

// Uniform buffer: глобальные параметры освещения
layout (binding = 2, std140) uniform LightingUniforms {
	vec3 ambient_color;  // Фоновый свет (ambient)
	uint num_lights;     // Количество активных источников
};

// Структура источника света 
struct LightSource {
	vec3 position;    // Позиция (для point и spot)
	float intensity;  // Яркость
	
	vec3 direction;   // Направление (для directional и spot)
	float range;      // Дальность действия (для затухания)
	
	vec3 color;       // Цвет света
	float cone_angle; // Угол конуса для spot (в градусах)
	
	uint type;        // 0=directional, 1=point, 2=spot
	uint enabled;     // Включён ли источник (0=выкл, 1=вкл)
	uint _pad[2];
};

// Storage buffer: массив всех источников света
layout (binding = 3, std430) readonly buffer LightsBuffer {
	LightSource lights[];
};

const float PI = 3.14159265359;

// Модель освещения Blinn-Phong
vec3 calculateLighting(vec3 normal, vec3 view_dir) {
	// Начинаем с ambient света - базовое освещение везде
	vec3 result = ambient_color * albedo_color;
	
	// Проходим по всем источникам света
	for (uint i = 0u; i < num_lights; ++i) {
		LightSource light = lights[i];
		
		// Пропускаем выключенные источники
		if (light.enabled == 0u) continue;
		
		vec3 light_dir;       // Направление к источнику
		float attenuation = 1.0;  // Коэффициент затухания
		
		if (light.type == 0u) {
			// Directional light (солнце) - параллельные лучи
			// Нет затухания, освещает всё одинаково
			light_dir = normalize(-light.direction);
			
		} else if (light.type == 1u) {
			// Point light (лампочка) - светит во все стороны
			vec3 light_vector = light.position - f_position;
			float distance = length(light_vector);
			light_dir = normalize(light_vector);
			
			// Затухание по закону обратных квадратов: 1 / (distance² / range²)
			// Чем дальше от источника, тем меньше attenuation
			attenuation = 1.0 / (1.0 + (distance * distance) / (light.range * light.range));
			
		} else if (light.type == 2u) {
			// Spot light (прожектор) - конус света
			vec3 light_vector = light.position - f_position;
			float distance = length(light_vector);
			light_dir = normalize(light_vector);
			
			// Проверяем попадание в конус света
			float cone_rad = light.cone_angle * PI / 180.0;
			float spot_cos = dot(light_dir, normalize(-light.direction));
			float spot_cutoff = cos(cone_rad);
			
			if (spot_cos < spot_cutoff) {
				// Вне конуса - нет света
				attenuation = 0.0;
			} else {
				// Внутри конуса - мягкие края + затухание
				attenuation = smoothstep(spot_cutoff, spot_cutoff + 0.1, spot_cos);
				attenuation *= 1.0 / (1.0 + (distance * distance) / (light.range * light.range));
			}
		}
		
		// Пропускаем источник если attenuation = 0
		if (attenuation <= 0.0) continue;
		
		// Diffuse компонент (Lambert) - зависит от угла между нормалью и светом
		// max(dot) чтобы не было отрицательных значений
		float diff = max(dot(normal, light_dir), 0.0);
		vec3 diffuse = diff * albedo_color;
		
		// Specular компонент (Blinn-Phong) - блики
		// Используем half-vector между light_dir и view_dir
		vec3 half_dir = normalize(light_dir + view_dir);
		float spec = pow(max(dot(normal, half_dir), 0.0), shininess);
		vec3 specular = spec * specular_color;
		
		// Итоговый вклад источника с учётом цвета, яркости и затухания
		vec3 light_contribution = (diffuse + specular) * light.color * light.intensity * attenuation;
		result += light_contribution;
	}
	
	return result;
}

void main() {
	vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(camera_position - f_position);  // Направление к камере
	
	vec3 color = calculateLighting(normal, view_dir);
	
	// Tone mapping - предотвращает пересвет (HDR -> LDR)
	color = color / (color + vec3(1.0));
	
	final_color = vec4(color, 1.0f);
}
