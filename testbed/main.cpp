#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

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
    };

    struct ModelUniforms
    {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0;
    };

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

    /*
     * Структура Material — представляет материал объекта (текстуру + сэмплер).
     * Каждый материал имеет свой descriptor set, что позволяет использовать
     * разные текстуры для разных моделей без перепривязки во время рендеринга.
     */
    struct Material
    {
        veekay::graphics::Texture *texture;   // VkImage + VkImageView + память
        VkSampler sampler;                    // Параметры сэмплирования (фильтрация, адресация)
        VkDescriptorSet descriptor_set;       // Набор дескрипторов для этого материала
    };

    /*
     * Model теперь содержит указатель на материал.
     * Разные модели могут использовать разные материалы (текстуры).
     */
    struct Model
    {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        Material *material;                   // Материал модели (текстура)
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
    }

    // NOTE: Vulkan objects
    inline namespace
    {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;   // Layout для set=0 (uniforms)
        VkDescriptorSetLayout material_set_layout;     // Layout для set=1 (текстуры материалов)
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer *scene_uniforms_buffer;
        veekay::graphics::Buffer *model_uniforms_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        // Материалы с разными текстурами для демонстрации мультитекстурирования
        Material *missing_material;   // Fallback текстура (шахматка) если файл не загрузился
        Material *lenna_material;     // Текстура из файла lenna.png
        Material *stone_material;     // Текстура каменной плитки для пола
        Material *gradient_material;  // Процедурно сгенерированный градиент
        Material *wood_material;      // Текстура дерева
    }

    float toRadians(float degrees)
    {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const
    {
        // TODO: Scaling and rotation

        auto t = veekay::mat4::translation(position);

        return t;
    }

    veekay::mat4 Camera::view() const
    {
        // TODO: Rotation

        auto t = veekay::mat4::translation(-position);

        return t;
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
                        // Увеличено количество COMBINED_IMAGE_SAMPLER для нескольких материалов
                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 16,
                    }};

                VkDescriptorPoolCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    // maxSets увеличен: 1 для uniforms + отдельный set для каждого материала
                    .maxSets = 16,
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

            // NOTE: Descriptor set layout specification
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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

            /*
             * Descriptor Set Layout для материалов (set = 1).
             * Содержит один combined image sampler — текстуру с сэмплером.
             * Каждый материал будет иметь свой descriptor set с этим layout.
             */
            {
                VkDescriptorSetLayoutBinding binding{
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,  // Только фрагментный шейдер
                };

                VkDescriptorSetLayoutCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .bindingCount = 1,
                    .pBindings = &binding,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                                &material_set_layout) != VK_SUCCESS)
                {
                    std::cerr << "Failed to create Vulkan material descriptor set layout\n";
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

            /*
             * Pipeline Layout использует два descriptor set layouts:
             * - Set 0: SceneUniforms + ModelUniforms (буферы)
             * - Set 1: Combined Image Sampler (текстура материала)
             */
            VkDescriptorSetLayout set_layouts[] = {
                descriptor_set_layout,
                material_set_layout,
            };

            VkPipelineLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,  // Два набора дескрипторов
                .pSetLayouts = set_layouts,
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

        scene_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(SceneUniforms),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
            max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        /*
         * Вспомогательная функция для создания материала.
         * Создаёт VkSampler с заданными параметрами фильтрации и адресации,
         * выделяет descriptor set и связывает его с текстурой.
         */
        auto createMaterial = [&](veekay::graphics::Texture *tex, VkFilter filter,
                                  VkSamplerAddressMode address_mode) -> Material *
        {
            Material *mat = new Material();
            mat->texture = tex;

            // Создание сэмплера с параметрами фильтрации и адресации текстурных координат
            VkSamplerCreateInfo sampler_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = filter,                          // Фильтр при увеличении
                .minFilter = filter,                          // Фильтр при уменьшении
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,  // Линейная интерполяция между mip-уровнями
                .addressModeU = address_mode,                 // Режим адресации по U
                .addressModeV = address_mode,                 // Режим адресации по V
                .addressModeW = address_mode,
                .mipLodBias = 0.0f,
                .anisotropyEnable = VK_TRUE,                  // Анизотропная фильтрация
                .maxAnisotropy = 16.0f,                       // Макс. уровень анизотропии
                .minLod = 0.0f,
                .maxLod = VK_LOD_CLAMP_NONE,                  // Использовать все mip-уровни
            };

            if (vkCreateSampler(device, &sampler_info, nullptr, &mat->sampler) != VK_SUCCESS)
            {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return nullptr;
            }

            // Выделение descriptor set для этого материала из пула
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &material_set_layout,  // Используем layout для материалов
            };

            if (vkAllocateDescriptorSets(device, &alloc_info, &mat->descriptor_set) != VK_SUCCESS)
            {
                std::cerr << "Failed to allocate material descriptor set\n";
                veekay::app.running = false;
                return nullptr;
            }

            // Запись в descriptor set: связываем binding 0 с текстурой и сэмплером
            VkDescriptorImageInfo image_info{
                .sampler = mat->sampler,
                .imageView = mat->texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkWriteDescriptorSet write_info{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat->descriptor_set,
                .dstBinding = 0,  // Соответствует layout(set=1, binding=0) в шейдере
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_info,
            };

            vkUpdateDescriptorSets(device, 1, &write_info, 0, nullptr);

            return mat;
        };

        // Fallback текстура 2x2 (чёрно-розовая шахматка) — используется если файл не загрузился
        {
            uint32_t pixels[] = {
                0xff000000,   // Чёрный
                0xffff00ff,   // Розовый (magenta)
                0xffff00ff,
                0xff000000,
            };

            auto *tex = new veekay::graphics::Texture(cmd, 2, 2,
                                                      VK_FORMAT_B8G8R8A8_UNORM,
                                                      pixels);
            // NEAREST фильтр чтобы пиксели были чёткими
            missing_material = createMaterial(tex, VK_FILTER_NEAREST,
                                              VK_SAMPLER_ADDRESS_MODE_REPEAT);
        }

        // Загрузка текстуры Lenna из PNG файла с помощью lodepng
        {
            std::vector<unsigned char> image_data;
            unsigned width, height;

            unsigned error = lodepng::decode(image_data, width, height, "./assets/lenna.png");
            if (error)
            {
                std::cerr << "Failed to load texture: " << lodepng_error_text(error) << "\n";
                lenna_material = missing_material;  // Fallback на missing текстуру
            }
            else
            {
                auto *tex = new veekay::graphics::Texture(cmd, width, height,
                                                          VK_FORMAT_R8G8B8A8_UNORM,
                                                          image_data.data());
                // LINEAR фильтр для плавного сглаживания
                lenna_material = createMaterial(tex, VK_FILTER_LINEAR,
                                                VK_SAMPLER_ADDRESS_MODE_REPEAT);
            }
        }

        // Загрузка текстуры каменной плитки для пола
        {
            std::vector<unsigned char> image_data;
            unsigned width, height;

            unsigned error = lodepng::decode(image_data, width, height, "./assets/stone.png");
            if (error)
            {
                std::cerr << "Failed to load stone texture: " << lodepng_error_text(error) << "\n";
                stone_material = missing_material;
            }
            else
            {
                auto *tex = new veekay::graphics::Texture(cmd, width, height,
                                                          VK_FORMAT_R8G8B8A8_UNORM,
                                                          image_data.data());
                // MIRRORED_REPEAT для бесшовного повторения текстуры
                stone_material = createMaterial(tex, VK_FILTER_LINEAR,
                                                VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
            }
        }

        // Загрузка текстуры дерева
        {
            std::vector<unsigned char> image_data;
            unsigned width, height;

            unsigned error = lodepng::decode(image_data, width, height, "./assets/wood.png");
            if (error)
            {
                std::cerr << "Failed to load wood texture: " << lodepng_error_text(error) << "\n";
                wood_material = missing_material;
            }
            else
            {
                auto *tex = new veekay::graphics::Texture(cmd, width, height,
                                                          VK_FORMAT_R8G8B8A8_UNORM,
                                                          image_data.data());
                wood_material = createMaterial(tex, VK_FILTER_LINEAR,
                                               VK_SAMPLER_ADDRESS_MODE_REPEAT);
            }
        }

        // Процедурно сгенерированная градиентная текстура (красно-зелёный градиент)
        {
            constexpr uint32_t size = 64;
            std::vector<uint32_t> pixels(size * size);

            for (uint32_t y = 0; y < size; ++y)
            {
                for (uint32_t x = 0; x < size; ++x)
                {
                    uint8_t r = (x * 255) / size;  // Красный увеличивается по X
                    uint8_t g = (y * 255) / size;  // Зелёный увеличивается по Y
                    uint8_t b = 128;
                    pixels[y * size + x] = (0xff << 24) | (b << 16) | (g << 8) | r;
                }
            }

            auto *tex = new veekay::graphics::Texture(cmd, size, size,
                                                      VK_FORMAT_R8G8B8A8_UNORM,
                                                      pixels.data());
            // CLAMP_TO_EDGE чтобы края не повторялись
            gradient_material = createMaterial(tex, VK_FILTER_LINEAR,
                                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        }

        {
            VkDescriptorBufferInfo buffer_infos[] = {
                {
                    .buffer = scene_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(SceneUniforms),
                },
                {
                    .buffer = model_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(ModelUniforms),
                },
            };

            VkWriteDescriptorSet write_infos[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[0],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_infos[1],
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
        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .material = stone_material, // Плоскость с каменной плиткой
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {-2.0f, -0.5f, -1.5f},
            },
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .material = lenna_material, // Куб с текстурой Lenna
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {1.5f, -0.5f, -0.5f},
            },
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .material = gradient_material, // Куб с градиентной текстурой
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {0.0f, -0.5f, 1.0f},
            },
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .material = wood_material, // Куб с текстурой дерева
        });
    }

    // NOTE: Destroy resources here, do not cause leaks in your program!
    void shutdown()
    {
        VkDevice &device = veekay::app.vk_device;

        // Helper to destroy material
        auto destroyMaterial = [&](Material *mat)
        {
            if (mat)
            {
                vkDestroySampler(device, mat->sampler, nullptr);
                delete mat->texture;
                delete mat;
            }
        };

        // Destroy materials (check for duplicates)
        if (gradient_material != missing_material)
        {
            destroyMaterial(gradient_material);
        }
        if (stone_material != missing_material)
        {
            destroyMaterial(stone_material);
        }
        if (wood_material != missing_material)
        {
            destroyMaterial(wood_material);
        }
        if (lenna_material != missing_material)
        {
            destroyMaterial(lenna_material);
        }
        destroyMaterial(missing_material);

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, material_set_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time)
    {
        ImGui::Begin("Controls:");
        ImGui::End();

        if (!ImGui::IsWindowHovered())
        {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::left))
            {
                auto move_delta = mouse::cursorDelta();

                // TODO: Use mouse_delta to update camera rotation

                auto view = camera.view();

                // TODO: Calculate right, up and front from view matrix
                veekay::vec3 right = {1.0f, 0.0f, 0.0f};
                veekay::vec3 up = {0.0f, -1.0f, 0.0f};
                veekay::vec3 front = {0.0f, 0.0f, 1.0f};

                if (keyboard::isKeyDown(keyboard::Key::w))
                    camera.position += front * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::s))
                    camera.position -= front * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::d))
                    camera.position += right * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::a))
                    camera.position -= right * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::q))
                    camera.position += up * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::z))
                    camera.position -= up * 0.1f;
            }
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i)
        {
            const Model &model = models[i];
            ModelUniforms &uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
        }

        *(SceneUniforms *)scene_uniforms_buffer->mapped_region = scene_uniforms;

        const size_t alignment =
            veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i)
        {
            const ModelUniforms &uniforms = model_uniforms[i];

            char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * alignment;
            *reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
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
        VkDescriptorSet current_material_set = VK_NULL_HANDLE;  // Кэш текущего материала

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

            // Привязка set 0 (uniforms) с динамическим смещением для ModelUniforms
            uint32_t offset = i * model_uniorms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set, 1, &offset);

            // Привязка set 1 (текстура материала) — только если материал изменился
            VkDescriptorSet material_set = model.material ? model.material->descriptor_set
                                                          : missing_material->descriptor_set;
            if (current_material_set != material_set)
            {
                current_material_set = material_set;
                // Привязываем descriptor set с текстурой к set=1 (без динамических смещений)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        1, 1, &current_material_set, 0, nullptr);
            }

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
