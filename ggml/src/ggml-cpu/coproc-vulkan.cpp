// CPU + GPU co-processing of Q4_0 matrix products, see coproc-vulkan.h

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "coproc-vulkan.h"
#include "coproc-q4_0.spv.h"  // generated from coproc-q4_0.comp at build time

#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

// must match coproc-q4_0.comp
constexpr int64_t COPROC_R  = 2;
constexpr int64_t COPROC_C  = 8;
constexpr int64_t COPROC_WG = 64;
constexpr int64_t COPROC_ROW_ALIGN = COPROC_R * COPROC_WG;  // rows per workgroup

#define COPROC_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { GGML_ABORT("%s failed: %d", #x, (int) r_); } } while (0)

struct coproc_buffer {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *         ptr = nullptr;
    size_t         size = 0;
};

struct coproc_stats {
    int64_t calls = 0;
    int64_t packed = 0;  // products whose activations were quantized (the others reused the previous ones)
    double  t_gpu = 0;   // submit to the end of the wait, as seen from the CPU
    double  t_wait = 0;  // time the CPU spent waiting for the fence
    double  t_pack = 0;  // activation quantization and packing
    double  t_copy = 0;  // copying the GPU's rows into dst
};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct coproc_ctx {
    bool ok = false;
    float fraction = 0.0f;
    int64_t min_n = 32;
    bool stats_enabled = false;
    coproc_stats stats;

    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    VkDevice         dev = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         queue_family = 0;
    uint32_t         mem_type = 0;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkCommandPool    cpool = VK_NULL_HANDLE;
    VkCommandBuffer  cb = VK_NULL_HANDLE;
    VkFence          fence = VK_NULL_HANDLE;
    uint32_t         dpool_left = 0;

    coproc_buffer act;  // packed activations, shared by all weights (one product at a time)
    coproc_buffer out;  // GPU result, shared by all weights

    // the activations last packed, so products that share src1 (q, k, v; gate, up) quantize them once
    const void * act_src = nullptr;
    int64_t act_ne0 = 0, act_ne1 = 0;
    uint64_t act_sample = 0;

    bool   need_pack = false;
    double t_setup = 0;
    double t_wait_end = 0;
    std::mutex mutex;
};

coproc_ctx & ctx() {
    static coproc_ctx c;
    return c;
}

}  // namespace

struct ggml_cpu_coproc_weight {
    const ggml_tensor * tensor;
    const void *        data;
    int64_t             M;       // rows of the weight
    int64_t             K;       // columns
    int64_t             rows;    // GPU rows: the last `rows` of the weight
    coproc_buffer       qa;      // nibbles, [NB][rows] x 16 bytes
    coproc_buffer       sa;      // scales, [NB][rows] floats
    VkDescriptorSet     ds = VK_NULL_HANDLE;
    bool                running = false;
    int64_t             n_cols = 0;
    double              t_submit = 0;
};

namespace {

std::unordered_map<const ggml_tensor *, std::unique_ptr<ggml_cpu_coproc_weight>> & weights() {
    static std::unordered_map<const ggml_tensor *, std::unique_ptr<ggml_cpu_coproc_weight>> w;
    return w;
}

coproc_buffer make_buffer(coproc_ctx & c, size_t size) {
    coproc_buffer b;
    b.size = size;
    VkBufferCreateInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    COPROC_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c.dev, b.buf, &mr);
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = c.mem_type;
    COPROC_CHECK(vkAllocateMemory(c.dev, &ai, nullptr, &b.mem));
    COPROC_CHECK(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
    COPROC_CHECK(vkMapMemory(c.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    return b;
}

void free_buffer(coproc_ctx & c, coproc_buffer & b) {
    if (b.buf) {
        vkUnmapMemory(c.dev, b.mem);
        vkDestroyBuffer(c.dev, b.buf, nullptr);
        vkFreeMemory(c.dev, b.mem, nullptr);
    }
    b = coproc_buffer();
}

void print_stats() {
    coproc_ctx & c = ctx();
    if (!c.stats_enabled || c.stats.calls == 0) {
        return;
    }
    fprintf(stderr, "coproc: %lld GPU products (%lld packed their activations), GPU %.3f s, CPU waiting for the GPU %.3f s, "
            "packing %.3f s, copying %.3f s\n", (long long) c.stats.calls, (long long) c.stats.packed, c.stats.t_gpu,
            c.stats.t_wait, c.stats.t_pack, c.stats.t_copy);
}

// lazy initialization on the first Q4_0 weight; false when the feature is off or no device is usable
bool init(coproc_ctx & c) {
    static bool tried = false;
    if (tried) {
        return c.ok;
    }
    tried = true;

    const char * f = getenv("GGML_CPU_COPROC");
    c.fraction = f ? (float) atof(f) : 0.0f;
    if (!(c.fraction > 0.0f)) {
        return false;
    }
    if (c.fraction > 0.9f) {
        c.fraction = 0.9f;
    }
    if (const char * n = getenv("GGML_CPU_COPROC_MIN_N")) {
        c.min_n = atoll(n);
    }
    c.stats_enabled = getenv("GGML_CPU_COPROC_STATS") && atoi(getenv("GGML_CPU_COPROC_STATS"));

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "ggml-cpu-coproc";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS) {
        GGML_LOG_WARN("%s: no Vulkan instance, co-processing off\n", __func__);
        return false;
    }
    uint32_t np = 0;
    vkEnumeratePhysicalDevices(c.instance, &np, nullptr);
    std::vector<VkPhysicalDevice> pds(np);
    vkEnumeratePhysicalDevices(c.instance, &np, pds.data());
    const char * dev_env = getenv("GGML_CPU_COPROC_DEVICE");
    for (uint32_t i = 0; i < np; i++) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(pds[i], &pp);
        if (dev_env ? (int) i == atoi(dev_env) : pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            VkPhysicalDeviceVulkan13Features f13 = {};
            f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            VkPhysicalDeviceFeatures2 f2 = {};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &f13;
            vkGetPhysicalDeviceFeatures2(pds[i], &f2);
            if (!f13.shaderIntegerDotProduct) {
                GGML_LOG_WARN("%s: %s has no integer dot product, skipped\n", __func__, pp.deviceName);
                continue;
            }
            c.pdev = pds[i];
            GGML_LOG_INFO("%s: co-processing on %s, %.0f%% of the rows of Q4_0 weights, products with >= %lld columns\n",
                          __func__, pp.deviceName, 100.0 * c.fraction, (long long) c.min_n);
            break;
        }
    }
    if (!c.pdev) {
        GGML_LOG_WARN("%s: no usable Vulkan device, co-processing off\n", __func__);
        return false;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.pdev, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qps(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.pdev, &nq, qps.data());
    c.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            c.queue_family = i;
            break;
        }
    }
    GGML_ASSERT(c.queue_family != UINT32_MAX);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13 = {};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.shaderIntegerDotProduct = VK_TRUE;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f13;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    COPROC_CHECK(vkCreateDevice(c.pdev, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.queue_family, 0, &c.queue);

    // host-visible memory: the CPU writes the weights and activations and reads the result directly
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(c.pdev, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    c.mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mp.memoryTypes[i].propertyFlags & want) == want) {
            // prefer cached memory: the CPU reads the result back
            if (c.mem_type == UINT32_MAX || (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                c.mem_type = i;
            }
        }
    }
    GGML_ASSERT(c.mem_type != UINT32_MAX);

    std::vector<uint32_t> code((coproc_q4_0_spv_len + 3) / 4);
    memcpy(code.data(), coproc_q4_0_spv, coproc_q4_0_spv_len);
    VkShaderModuleCreateInfo smi = {};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = coproc_q4_0_spv_len;
    smi.pCode = code.data();
    VkShaderModule sm;
    COPROC_CHECK(vkCreateShaderModule(c.dev, &smi, nullptr, &sm));
    VkDescriptorSetLayoutBinding lb[4];
    for (uint32_t i = 0; i < 4; i++) {
        lb[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    VkDescriptorSetLayoutCreateInfo dli = {};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 4;
    dli.pBindings = lb;
    COPROC_CHECK(vkCreateDescriptorSetLayout(c.dev, &dli, nullptr, &c.dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 };
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    COPROC_CHECK(vkCreatePipelineLayout(c.dev, &pli, nullptr, &c.pl));
    VkComputePipelineCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
    cpi.layout = c.pl;
    COPROC_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &c.pipeline));
    vkDestroyShaderModule(c.dev, sm, nullptr);

    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.queue_family;
    COPROC_CHECK(vkCreateCommandPool(c.dev, &cpci, nullptr, &c.cpool));
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = c.cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    COPROC_CHECK(vkAllocateCommandBuffers(c.dev, &cbai, &c.cb));
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    COPROC_CHECK(vkCreateFence(c.dev, &fci, nullptr, &c.fence));

    atexit(print_stats);
    c.ok = true;
    return true;
}

VkDescriptorSet alloc_set(coproc_ctx & c) {
    if (c.dpool_left == 0) {
        const uint32_t n = 256;
        VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * n };
        VkDescriptorPoolCreateInfo dpi = {};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = n;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        COPROC_CHECK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &c.dpool));  // pools are kept for the process lifetime
        c.dpool_left = n;
    }
    VkDescriptorSetAllocateInfo dsa = {};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = c.dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &c.dsl;
    VkDescriptorSet ds;
    COPROC_CHECK(vkAllocateDescriptorSets(c.dev, &dsa, &ds));
    c.dpool_left--;
    return ds;
}

uint64_t sample_hash(const float * x, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    const int64_t step = n > 64 ? n / 64 : 1;
    for (int64_t i = 0; i < n; i += step) {
        uint32_t u;
        memcpy(&u, x + i, 4);
        h = (h ^ u) * 1099511628211ull;
    }
    return h;
}

}  // namespace

void ggml_cpu_coproc_prepare(const struct ggml_tensor * t, const void * data) {
    coproc_ctx & c = ctx();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (t->type != GGML_TYPE_Q4_0 || !init(c)) {
        return;
    }
    if (ggml_n_dims(t) != 2 || t->ne[0] % QK4_0 != 0) {
        return;
    }
    const int64_t K = t->ne[0], M = t->ne[1], NB = K / QK4_0;
    const int64_t rows = (int64_t) (c.fraction * (float) M) / COPROC_ROW_ALIGN * COPROC_ROW_ALIGN;
    if (rows <= 0) {
        return;
    }

    auto w = std::make_unique<ggml_cpu_coproc_weight>();
    w->tensor = t;
    w->data = t->data;
    w->M = M;
    w->K = K;
    w->rows = rows;
    w->qa = make_buffer(c, (size_t) NB * rows * 16);
    w->sa = make_buffer(c, (size_t) NB * rows * sizeof(float));

    const block_q4_0 * src = (const block_q4_0 *) data;
    uint8_t * qa = (uint8_t *) w->qa.ptr;
    float * sa = (float *) w->sa.ptr;
    const int64_t r_first = M - rows;
    std::vector<uint8_t> tq((size_t) NB * rows * 16);  // staged in normal memory: the mapping is uncached on some devices
    std::vector<float> ts((size_t) NB * rows);
    for (int64_t r = 0; r < rows; r++) {
        const block_q4_0 * row = src + (r_first + r) * NB;
        for (int64_t j = 0; j < NB; j++) {
            memcpy(&tq[((size_t) j * rows + r) * 16], row[j].qs, 16);
            ts[(size_t) j * rows + r] = ggml_fp16_to_fp32(row[j].d);
        }
    }
    memcpy(qa, tq.data(), tq.size());
    memcpy(sa, ts.data(), ts.size() * sizeof(float));

    w->ds = alloc_set(c);
    weights()[t] = std::move(w);
}

struct ggml_cpu_coproc_weight * ggml_cpu_coproc_find(const struct ggml_tensor * t) {
    coproc_ctx & c = ctx();
    if (!c.ok) {
        return nullptr;
    }
    auto it = weights().find(t);
    if (it == weights().end() || it->second->data != t->data) {
        return nullptr;
    }
    return it->second.get();
}

int64_t ggml_cpu_coproc_rows(const struct ggml_cpu_coproc_weight * w, int64_t n_cols) {
    return w && n_cols >= ctx().min_n ? w->rows : 0;
}

void ggml_cpu_coproc_setup(struct ggml_cpu_coproc_weight * w, const struct ggml_tensor * src1) {
    coproc_ctx & c = ctx();
    c.t_setup = now_s();

    const int64_t K = w->K, NB = K / QK8_1, N = src1->ne[1];
    const int64_t Np = (N + COPROC_C - 1) / COPROC_C * COPROC_C;
    GGML_ASSERT(src1->ne[0] == K && src1->type == GGML_TYPE_F32);

    const size_t act_size = (size_t) Np * NB * sizeof(block_q8_1);
    const size_t out_size = (size_t) Np * w->rows * sizeof(float);
    if (c.act.size < act_size) {
        free_buffer(c, c.act);
        c.act = make_buffer(c, act_size + act_size / 4);
        c.act_src = nullptr;
    }
    if (c.out.size < out_size) {
        free_buffer(c, c.out);
        c.out = make_buffer(c, out_size + out_size / 4);
    }

    // the activations are packed again unless this src1 was the last one packed (q, k, v share it; so do gate, up)
    const uint64_t sample = sample_hash((const float *) src1->data, K) ^
                            (sample_hash((const float *) ((const char *) src1->data + (N - 1) * src1->nb[1]), K) << 1);
    c.need_pack = c.act_src != src1->data || c.act_ne0 != K || c.act_ne1 != N || c.act_sample != sample;
    if (c.need_pack) {
        c.stats.packed++;
        c.act_src = src1->data;
        c.act_ne0 = K;
        c.act_ne1 = N;
        c.act_sample = sample;
    }
    w->n_cols = N;
}

void ggml_cpu_coproc_pack(struct ggml_cpu_coproc_weight * w, const struct ggml_tensor * src1, int ith, int nth) {
    coproc_ctx & c = ctx();
    if (!c.need_pack) {
        return;
    }
    // quantize the activations to Q8_1 (d, d * sum, 32 int8 per block) into the GPU's layout
    // [N / C][NB][C][block_q8_1], columns past N zeroed; thread ith packs columns ith, ith + nth, ...
    const int64_t K = w->K, NB = K / QK8_1, N = w->n_cols;
    const int64_t Np = (N + COPROC_C - 1) / COPROC_C * COPROC_C;
    const ggml_from_float_t quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_1)->from_float;
    block_q8_1 * dst = (block_q8_1 *) c.act.ptr;
    thread_local std::vector<block_q8_1> row;
    row.resize(NB);
    for (int64_t col = ith; col < Np; col += nth) {
        const int64_t g = col / COPROC_C, cc = col % COPROC_C;
        if (col < N) {
            quantize((const float *) ((const char *) src1->data + col * src1->nb[1]), row.data(), K);
        } else {
            memset(row.data(), 0, NB * sizeof(block_q8_1));
        }
        for (int64_t j = 0; j < NB; j++) {
            dst[(g * NB + j) * COPROC_C + cc] = row[j];
        }
    }
}

void ggml_cpu_coproc_submit(struct ggml_cpu_coproc_weight * w) {
    coproc_ctx & c = ctx();
    const int64_t NB = w->K / QK8_1;
    const int64_t Np = (w->n_cols + COPROC_C - 1) / COPROC_C * COPROC_C;
    const double t1 = now_s();

    VkDescriptorBufferInfo dbi[4] = {
        { w->qa.buf, 0, VK_WHOLE_SIZE },
        { w->sa.buf, 0, VK_WHOLE_SIZE },
        { c.act.buf, 0, VK_WHOLE_SIZE },
        { c.out.buf, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet wds[4];
    for (uint32_t i = 0; i < 4; i++) {
        wds[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, w->ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi[i], nullptr };
    }
    vkUpdateDescriptorSets(c.dev, 4, wds, 0, nullptr);

    COPROC_CHECK(vkResetCommandBuffer(c.cb, 0));
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    COPROC_CHECK(vkBeginCommandBuffer(c.cb, &cbbi));
    vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
    vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &w->ds, 0, nullptr);
    // one dispatch: x = row blocks, y = column groups (a small share of rows still fills the GPU)
    const uint32_t pc[3] = { (uint32_t) w->rows, (uint32_t) Np, (uint32_t) NB };
    vkCmdPushConstants(c.cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(c.cb, (uint32_t) (w->rows / COPROC_ROW_ALIGN), (uint32_t) (Np / COPROC_C), 1);
    COPROC_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cb;
    COPROC_CHECK(vkQueueSubmit(c.queue, 1, &si, c.fence));

    w->running = true;
    w->t_submit = now_s();
    c.stats.t_pack += t1 - c.t_setup;
}

void ggml_cpu_coproc_wait(struct ggml_cpu_coproc_weight * w) {
    coproc_ctx & c = ctx();
    GGML_ASSERT(w->running);
    const double t0 = now_s();
    COPROC_CHECK(vkWaitForFences(c.dev, 1, &c.fence, VK_TRUE, UINT64_MAX));
    COPROC_CHECK(vkResetFences(c.dev, 1, &c.fence));
    const double t1 = now_s();
    w->running = false;
    c.stats.calls++;
    c.stats.t_gpu += t1 - w->t_submit;
    c.stats.t_wait += t1 - t0;
    c.t_wait_end = t1;
}

void ggml_cpu_coproc_copy(struct ggml_cpu_coproc_weight * w, struct ggml_tensor * dst, int ith, int nth) {
    coproc_ctx & c = ctx();
    // y is [Np][rows]; dst column col holds rows [M - rows, M) of the product; thread ith copies columns ith, ith + nth, ...
    const float * y = (const float *) c.out.ptr;
    for (int64_t col = ith; col < w->n_cols; col += nth) {
        float * d = (float *) ((char *) dst->data + col * dst->nb[1]) + (w->M - w->rows);
        memcpy(d, y + col * w->rows, w->rows * sizeof(float));
    }
    if (ith == 0) {
        c.stats.t_copy += now_s() - c.t_wait_end;
    }
}
