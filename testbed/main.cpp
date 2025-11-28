#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>

namespace
{

    constexpr uint32_t max_models = 1024;

    struct Vertex
    {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
        // NOTE: You can add more attributes
    };

    struct SceneUniforms
    {
        veekay::mat4 view_projection;
        veekay::vec3 camera_position;
        float _pad0; //
    };

    // Uniforms для каждой модели (трансформация + материал)
    struct ModelUniforms
    {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0; // Диффузный цвет (основной цвет поверхности)
        veekay::vec3 specular_color;
        float shininess; // Цвет бликов и их резкость (2-256)
    };

    // Источник света для передачи в шейдер через storage buffer
    // Поддерживает 3 типа: directional (солнце), point (лампочка), spot (фонарик)
    struct LightSource
    {
        veekay::vec3 position; // Позиция источника (для point и spot)
        float intensity;       // Яркость света

        veekay::vec3 direction; // Направление света (для directional и spot)
        float range;            // Дальность действия света (для затухания)

        veekay::vec3 color; // Цвет света (RGB)
        float cone_angle;   // Угол конуса в градусах (только для spot)

        uint32_t type;    // Тип: 0=directional, 1=point, 2=spot
        uint32_t enabled; // Включён ли источник света (0=выкл, 1=вкл)
        uint32_t _pad[2]; // Padding для выравнивания в std430
    };

    // Глобальные параметры освещения сцены
    struct LightingUniforms
    {
        veekay::vec3 ambient_color; // Ambient свет (базовое освещение везде)
        uint32_t num_lights;        // Количество активных источников
    };

    constexpr uint32_t max_lights = 256;

    struct Mesh
    {
        veekay::graphics::Buffer *vertex_buffer;
        veekay::graphics::Buffer *index_buffer;
        uint32_t indices;
    };

    struct Transform
    {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        // NOTE: Model matrix (translation, rotation and scaling)
        veekay::mat4 matrix() const;
    };

    // Модель с материалом Blinn-Phong
    struct Model
    {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;   // Диффузный цвет (основной цвет материала)
        veekay::vec3 specular_color; // Цвет бликов (обычно белый/серый)
        float shininess;             // Резкость бликов (2=размыто, 256=острые)
    };

    struct Camera
    {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {};

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        // NOTE: View matrix of camera (inverse of a transform)
        veekay::mat4 view() const;

        // NOTE: View and projection composition
        veekay::mat4 view_projection(float aspect_ratio) const;
    };

    // NOTE: Scene objects
    inline namespace
    {
        Camera camera{
            .position = {0.0f, -0.5f, -3.0f}};

        std::vector<Model> models;

        // Освещение сцены
        veekay::vec3 ambient_color = {0.2f, 0.2f, 0.2f}; // Фоновый свет
        std::vector<LightSource> lights;                 // Все источники света
    }

    // NOTE: Vulkan objects
    inline namespace
    {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        // Буферы для передачи данных в шейдеры
        veekay::graphics::Buffer *scene_uniforms_buffer;    // Камера и проекция
        veekay::graphics::Buffer *model_uniforms_buffer;    // Трансформации и материалы моделей
        veekay::graphics::Buffer *lighting_uniforms_buffer; // Ambient и количество источников
        veekay::graphics::Buffer *lights_storage_buffer;    // Storage buffer: массив всех источников света

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture *missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture *texture;
        VkSampler texture_sampler;
    }

    float toRadians(float degrees)
    {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const
    {
        // Создаем матрицы трансформации: порядок T * R * S
        auto t = veekay::mat4::translation(position);

        // Создаем матрицы поворота для каждой оси (Euler angles: XYZ)
        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
        auto r = ry * rx * rz; // Порядок применения поворотов: Y, X, Z

        auto s = veekay::mat4::scaling(scale);

        // Применяем трансформации: сначала масштаб, затем поворот, затем сдвиг
        return t * r * s;
    }

    veekay::mat4 Camera::view() const
    {
        float yaw_rad = toRadians(rotation.y);
        float pitch_rad = toRadians(rotation.x);
        veekay::mat4 rot_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, yaw_rad);
        veekay::mat4 rot_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, pitch_rad);
        veekay::mat4 rot = rot_y * rot_x;

        veekay::mat4 rot_t = veekay::mat4::transpose(rot);

        veekay::vec3 neg_pos = -position;
        veekay::mat4 t = veekay::mat4::translation(neg_pos);

        return rot_t * t;
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const
    {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

        return view() * projection;
    }

    // NOTE: Loads shader byte code from file
    // NOTE: Your shaders are compiled via CMake with this code too, look it up
    VkShaderModule loadShaderModule(const char *path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = buffer.data(),
        };

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS)
        {
            return nullptr;
        }

        return result;
    }

    void initialize(VkCommandBuffer cmd)
    {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        { // NOTE: Build graphics pipeline
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (!vertex_shader_module)
            {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (!fragment_shader_module)
            {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            VkPipelineShaderStageCreateInfo stage_infos[2];

            // NOTE: Vertex shader stage
            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertex_shader_module,
                .pName = "main",
            };

            // NOTE: Fragment shader stage
            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragment_shader_module,
                .pName = "main",
            };

            // NOTE: How many bytes does a vertex take?
            VkVertexInputBindingDescription buffer_binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            // NOTE: Declare vertex attributes
            VkVertexInputAttributeDescription attributes[] = {
                {
                    .location = 0,                        // NOTE: First attribute
                    .binding = 0,                         // NOTE: First vertex buffer
                    .format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
                    .offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal),
                },
                {
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = offsetof(Vertex, uv),
                },
            };

            // NOTE: Describe inputs
            VkPipelineVertexInputStateCreateInfo input_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &buffer_binding,
                .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                .pVertexAttributeDescriptions = attributes,
            };

            // NOTE: Every three vertices make up a triangle,
            //       so our vertex buffer contains a "list of triangles"
            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            // NOTE: Declare clockwise triangle order as front-facing
            //       Discard triangles that are facing away
            //       Fill triangles, don't draw lines instaed
            VkPipelineRasterizationStateCreateInfo raster_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .lineWidth = 1.0f,
            };

            // NOTE: Use 1 sample per pixel
            VkPipelineMultisampleStateCreateInfo sample_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = false,
                .minSampleShading = 1.0f,
            };

            VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(veekay::app.window_width),
                .height = static_cast<float>(veekay::app.window_height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                .offset = {0, 0},
                .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            // NOTE: Let rasterizer draw on the entire window
            VkPipelineViewportStateCreateInfo viewport_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

                .viewportCount = 1,
                .pViewports = &viewport,

                .scissorCount = 1,
                .pScissors = &scissor,
            };

            // NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
            VkPipelineDepthStencilStateCreateInfo depth_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = true,
                .depthWriteEnable = true,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            // NOTE: Let fragment shader write all the color channels
            VkPipelineColorBlendAttachmentState attachment_info{
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                  VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT,
            };

            // NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
            VkPipelineColorBlendStateCreateInfo blend_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

                .logicOpEnable = false,
                .logicOp = VK_LOGIC_OP_COPY,

                .attachmentCount = 1,
                .pAttachments = &attachment_info};

            {
                VkDescriptorPoolSize pools[] = {
                    {
                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 8,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 8,
                    },
                    {
                        // Storage buffer - для массива источников света
                        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 8,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 8,
                    }};

                VkDescriptorPoolCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    .maxSets = 1,
                    .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                    .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr,
                                           &descriptor_pool) != VK_SUCCESS)
                {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Descriptor set layout - описываем какие буферы будут в шейдерах
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0, // SceneUniforms: view_projection + camera_position
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 1, // ModelUniforms: model matrix + материал (dynamic - разный для каждой модели)
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 2, // LightingUniforms: ambient цвет + количество источников
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 3, // Storage buffer: массив всех источников света
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                };

                VkDescriptorSetLayoutCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                    .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                                &descriptor_set_layout) != VK_SUCCESS)
                {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetAllocateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &descriptor_set_layout,
                };

                if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS)
                {
                    std::cerr << "Failed to create Vulkan descriptor set\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // NOTE: Declare external data sources, only push constants this time
            VkPipelineLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };

            // NOTE: Create pipeline layout
            if (vkCreatePipelineLayout(device, &layout_info,
                                       nullptr, &pipeline_layout) != VK_SUCCESS)
            {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }

            VkGraphicsPipelineCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = stage_infos,
                .pVertexInputState = &input_state_info,
                .pInputAssemblyState = &assembly_state_info,
                .pViewportState = &viewport_info,
                .pRasterizationState = &raster_info,
                .pMultisampleState = &sample_info,
                .pDepthStencilState = &depth_info,
                .pColorBlendState = &blend_info,
                .layout = pipeline_layout,
                .renderPass = veekay::app.vk_render_pass,
            };

            // NOTE: Create graphics pipeline
            if (vkCreateGraphicsPipelines(device, nullptr,
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS)
            {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        // Создание буферов для передачи данных в шейдеры
        scene_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(SceneUniforms),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
            max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // Буфер для глобальных параметров освещения
        lighting_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(LightingUniforms),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // Storage buffer - позволяет передать массив динамического размера в шейдер
        lights_storage_buffer = new veekay::graphics::Buffer(
            max_lights * sizeof(LightSource),
            nullptr,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // NOTE: This texture and sampler is used when texture could not be loaded
        {
            VkSamplerCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };

            if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS)
            {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return;
            }

            uint32_t pixels[] = {
                0xff000000,
                0xffff00ff,
                0xffff00ff,
                0xff000000,
            };

            missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                            VK_FORMAT_B8G8R8A8_UNORM,
                                                            pixels);
        }

        {
            // Описания буферов должны жить до вызова vkUpdateDescriptorSets
            VkDescriptorBufferInfo buffer_info_0{
                .buffer = scene_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(SceneUniforms),
            };

            VkDescriptorBufferInfo buffer_info_1{
                .buffer = model_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            };

            VkDescriptorBufferInfo buffer_info_2{
                // Униформы освещения
                .buffer = lighting_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(LightingUniforms),
            };

            VkDescriptorBufferInfo storage_buffer_info{
                // Массив источников света
                .buffer = lights_storage_buffer->buffer,
                .offset = 0,
                .range = max_lights * sizeof(LightSource),
            };

            VkWriteDescriptorSet write_infos[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 0, // SceneUniforms
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_info_0,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 1, // ModelUniforms (dynamic)
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_info_1,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 2, // LightingUniforms
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_info_2,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 3, // Storage buffer для источников
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &storage_buffer_info,
                },
            };

            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }

        // NOTE: Plane mesh initialization
        {
            // (v0)------(v1)
            //  |  \       |
            //  |   `--,   |
            //  |       \  |
            // (v3)------(v2)
            std::vector<Vertex> vertices = {
                {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0};

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            plane_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Cube mesh initialization
        {
            std::vector<Vertex> vertices = {
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

                {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

                {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

                {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                0,
                1,
                2,
                2,
                3,
                0,
                4,
                5,
                6,
                6,
                7,
                4,
                8,
                9,
                10,
                10,
                11,
                8,
                12,
                13,
                14,
                14,
                15,
                12,
                16,
                17,
                18,
                18,
                19,
                16,
                20,
                21,
                22,
                22,
                23,
                20,
            };

            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            cube_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Add models to scene
        // Плоскость
        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .albedo_color = veekay::vec3{0.8f, 0.8f, 0.8f},
            .specular_color = veekay::vec3{0.2f, 0.2f, 0.2f},
            .shininess = 8.0f});

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {-2.0f, -0.5f, -1.5f},
            },
            .albedo_color = veekay::vec3{0.8f, 0.1f, 0.1f},
            .specular_color = veekay::vec3{1.0f, 0.8f, 0.8f},
            .shininess = 32.0f});

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {1.5f, -0.5f, -0.5f},
            },
            .albedo_color = veekay::vec3{0.1f, 0.8f, 0.1f},
            .specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .shininess = 128.0f});

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {0.0f, -0.5f, 1.0f},
            },
            .albedo_color = veekay::vec3{0.2f, 0.2f, 0.6f},
            .specular_color = veekay::vec3{0.9f, 0.9f, 1.0f},
            .shininess = 64.0f});

        // Настраиваем источники света с разными цветами для наглядности

        // 1. Directional light - как солнце, освещает всё одинаково
        lights.emplace_back(LightSource{
            .position = {0.0f, 0.0f, 0.0f}, // Не используется
            .intensity = 0.5f,
            .direction = {-0.3f, -1.0f, -0.2f},
            .range = 100.0f,
            .color = {1.0f, 0.95f, 0.8f},
            .cone_angle = 0.0f,
            .type = 0, // directional
            .enabled = 1,
        });

        // 2. Point light - как лампочка, затухает с расстоянием
        lights.emplace_back(LightSource{
            .position = {-4.0f, 2.0f, 0.0f},
            .intensity = 1.0f,
            .direction = {0.0f, -1.0f, 0.0f}, // Не используется
            .range = 12.0f,
            .color = {1.0f, 0.3f, 0.2f},
            .cone_angle = 0.0f,
            .type = 1, // point
            .enabled = 1,
        });

        // 3. Spot light - как фонарик, светит конусом
        lights.emplace_back(LightSource{
            .position = {4.0f, 3.0f, 2.0f},
            .intensity = 1.2f,
            .direction = {-0.6f, -0.5f, -0.4f},
            .range = 15.0f,
            .color = {0.2f, 0.5f, 1.0f},
            .cone_angle = 35.0f,
            .type = 2, // spot
            .enabled = 1,
        });
    }

    // NOTE: Destroy resources here, do not cause leaks in your program!
    void shutdown()
    {
        VkDevice &device = veekay::app.vk_device;

        vkDestroySampler(device, missing_texture_sampler, nullptr);
        delete missing_texture;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete lights_storage_buffer;
        delete lighting_uniforms_buffer;
        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time)
    {
        using namespace veekay::input;

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 700), ImGuiCond_FirstUseEver);
        ImGui::Begin("Camera & Lighting Controls");

        ImGui::SeparatorText("Camera");
        ImGui::Text("Position: %.2f, %.2f, %.2f", camera.position.x, camera.position.y, camera.position.z);
        ImGui::Text("Rotation: %.2f, %.2f", camera.rotation.x, camera.rotation.y);
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Controls:");
        ImGui::BulletText("LMB + Drag: Rotate camera");
        ImGui::BulletText("WASD: Move horizontally");
        ImGui::BulletText("Q/Z: Move up/down");

        ImGui::SeparatorText("Ambient Light");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Base lighting everywhere (no direction)");
        ImGui::ColorEdit3("Ambient Color##1", &ambient_color.x);
        ImGui::Text("Try setting to (0,0,0) to see only direct lights!");

        ImGui::SeparatorText("Light Sources");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Total lights: %d", (int)lights.size());

        for (size_t i = 0; i < lights.size(); ++i)
        {
            LightSource &light = lights[i];

            const char *light_types[] = {"Directional", "Point", "Spot"};

            std::string light_label = std::string("Light ") + std::to_string(i) + " (" + light_types[light.type] + ")";
            if (ImGui::CollapsingHeader(light_label.c_str()))
            {
                std::string prefix = "##light" + std::to_string(i);

                bool enabled = light.enabled != 0;
                if (ImGui::Checkbox(("Enabled" + prefix).c_str(), &enabled))
                {
                    light.enabled = enabled ? 1 : 0;
                }

                ImGui::Combo(("Type" + prefix).c_str(), (int *)&light.type, light_types, 3);

                if (light.type == 0)
                {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Like sun: parallel rays, no position");
                }
                else if (light.type == 1)
                {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Like bulb: omnidirectional, attenuates");
                }
                else
                {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Like flashlight: cone, position + direction");
                }

                ImGui::ColorEdit3(("Color" + prefix).c_str(), &light.color.x);
                ImGui::SliderFloat(("Intensity" + prefix).c_str(), &light.intensity, 0.0f, 2.0f);

                if (light.type != 0)
                { // Not directional
                    ImGui::Text("Position & Range:");
                    ImGui::DragFloat3(("Position" + prefix).c_str(), &light.position.x, 0.1f);
                    ImGui::SliderFloat(("Range" + prefix).c_str(), &light.range, 1.0f, 50.0f);
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Range affects attenuation (inverse square law)");
                }

                if (light.type == 0 || light.type == 2)
                {
                    ImGui::Text("Direction:");
                    ImGui::DragFloat3(("Direction" + prefix).c_str(), &light.direction.x, 0.1f);
                    float length = std::sqrt(light.direction.x * light.direction.x +
                                             light.direction.y * light.direction.y +
                                             light.direction.z * light.direction.z);
                    if (length > 0.001f)
                    {
                        light.direction.x /= length;
                        light.direction.y /= length;
                        light.direction.z /= length;
                    }
                }

                if (light.type == 2)
                {
                    ImGui::Text("Cone Settings:");
                    ImGui::SliderFloat(("Cone Angle" + prefix).c_str(), &light.cone_angle, 5.0f, 90.0f);
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Angle of the light cone in degrees");
                }
            }
        }

        ImGui::SeparatorText("Model Materials");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Blinn-Phong material properties");
        for (size_t i = 0; i < models.size(); ++i)
        {
            Model &model = models[i];
            std::string model_label = std::string("Model ") + std::to_string(i);

            if (ImGui::CollapsingHeader(model_label.c_str()))
            {
                std::string prefix = "##model" + std::to_string(i);
                ImGui::ColorEdit3(("Albedo (Diffuse)" + prefix).c_str(), &model.albedo_color.x);
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Base color of the surface");

                ImGui::ColorEdit3(("Specular" + prefix).c_str(), &model.specular_color.x);
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Color of reflective highlights");

                ImGui::SliderFloat(("Shininess" + prefix).c_str(), &model.shininess, 2.0f, 256.0f);
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "How sharp the specular highlight is");
            }
        }

        ImGui::End();

        if (mouse::isButtonDown(mouse::Button::left) && !ImGui::GetIO().WantCaptureMouse)
        {
            auto move_delta = mouse::cursorDelta();

            float sensitivity = 0.1f;
            camera.rotation.y -= move_delta.x * sensitivity;
            camera.rotation.x -= move_delta.y * sensitivity;
            camera.rotation.x = std::clamp(camera.rotation.x, -89.0f, 89.0f);
            camera.rotation.z = 0.0f;
        }

        // Вычисляем направления камеры на основе её вращения
        float yaw_rad = toRadians(camera.rotation.y);

        // Forward и Right для горизонтального движения (без учета pitch)
        veekay::vec3 forward = {
            -std::sin(yaw_rad),
            0.0f,
            std::cos(yaw_rad)};

        veekay::vec3 right = {
            std::cos(yaw_rad),
            0.0f,
            std::sin(yaw_rad)};

        veekay::vec3 up = {0.0f, 1.0f, 0.0f};

        if (!ImGui::GetIO().WantCaptureKeyboard)
        {
            float speed = 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::w))
                camera.position += forward * speed;

            if (keyboard::isKeyDown(keyboard::Key::s))
                camera.position -= forward * speed;

            if (keyboard::isKeyDown(keyboard::Key::d))
                camera.position += right * speed;

            if (keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= right * speed;

            if (keyboard::isKeyDown(keyboard::Key::q))
                camera.position -= up * speed;

            if (keyboard::isKeyDown(keyboard::Key::z))
                camera.position += up * speed;
        }

        // Обновляем uniform buffers для передачи в шейдеры
        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .camera_position = camera.position, // Нужна для вычисления specular бликов
        };

        // Подготавливаем данные для каждой модели
        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i)
        {
            const Model &model = models[i];
            ModelUniforms &uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
            uniforms.specular_color = model.specular_color;
            uniforms.shininess = model.shininess;
        }

        // Обновляем параметры освещения
        LightingUniforms lighting_uniforms{
            .ambient_color = ambient_color,
            .num_lights = static_cast<uint32_t>(lights.size()),
        };

        // Копируем данные в GPU буферы
        *(SceneUniforms *)scene_uniforms_buffer->mapped_region = scene_uniforms;
        *(LightingUniforms *)lighting_uniforms_buffer->mapped_region = lighting_uniforms;

        // Копируем данные моделей с учётом alignment для dynamic uniform buffer
        const size_t alignment =
            veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i)
        {
            const ModelUniforms &uniforms = model_uniforms[i];

            char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * alignment;
            *reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
        }

        // Копируем массив источников света в storage buffer
        for (size_t i = 0; i < lights.size(); ++i)
        {
            LightSource *lights_ptr = static_cast<LightSource *>(lights_storage_buffer->mapped_region);
            lights_ptr[i] = lights[i];
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer)
    {
        vkResetCommandBuffer(cmd, 0);

        { // NOTE: Start recording rendering commands
            VkCommandBufferBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }

        { // NOTE: Use current swapchain framebuffer and clear it
            VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

            VkClearValue clear_values[] = {clear_color, clear_depth};

            VkRenderPassBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = veekay::app.vk_render_pass,
                .framebuffer = framebuffer,
                .renderArea = {
                    .extent = {
                        veekay::app.window_width,
                        veekay::app.window_height},
                },
                .clearValueCount = 2,
                .pClearValues = clear_values,
            };

            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize zero_offset = 0;

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        const size_t model_uniorms_alignment =
            veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = models.size(); i < n; ++i)
        {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer)
            {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer)
            {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = i * model_uniorms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace

int main()
{
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}