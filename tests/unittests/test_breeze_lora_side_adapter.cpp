// Verifies the Breeze side-adapter execution path on a real ggml graph:
// build_adapter_linear must produce base(x) + (alpha/rank) * B(A(x)), and the
// disabled (null-adapter) path must be exactly base(x).
#include "engine/framework/core/backend.h"
#include "engine/models/breeze_tts/lora_linear.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ggml.h>

using engine::models::breeze_tts::BreezeLoraLinearContext;
using engine::models::breeze_tts::build_adapter_linear;
using engine::models::breeze_tts::kBreezeLoraRank;
using engine::modules::LinearWeights;
using engine::test::require;

namespace {

constexpr int64_t kTestGraphBytes = 4 * 1024 * 1024;
constexpr int kTestGraphNodes = 4096;

bool nearly_equal(float a, float b, float eps = 1e-4F) {
    return std::fabs(a - b) <= eps;
}

// Reference: base(x) + scale * B(A(x)); A is [rank,in], B is [out,rank].
std::vector<float> reference(
    const std::vector<float> & w,
    const std::vector<float> & a,
    const std::vector<float> & b,
    float scale,
    bool enabled,
    int64_t in_f,
    int64_t out_f,
    int64_t rank,
    const std::vector<float> & x) {
    std::vector<float> out(static_cast<size_t>(out_f), 0.0F);
    for (int64_t o = 0; o < out_f; ++o) {
        float s = 0.0F;
        for (int64_t i = 0; i < in_f; ++i) {
            s += x[static_cast<size_t>(i)] * w[static_cast<size_t>(o * in_f + i)];
        }
        out[static_cast<size_t>(o)] = s;
    }
    if (!enabled) {
        return out;
    }
    std::vector<float> mid(static_cast<size_t>(rank), 0.0F);
    for (int64_t r = 0; r < rank; ++r) {
        float s = 0.0F;
        for (int64_t i = 0; i < in_f; ++i) {
            s += x[static_cast<size_t>(i)] * a[static_cast<size_t>(r * in_f + i)];
        }
        mid[static_cast<size_t>(r)] = s;
    }
    for (int64_t o = 0; o < out_f; ++o) {
        float s = 0.0F;
        for (int64_t r = 0; r < rank; ++r) {
            s += mid[static_cast<size_t>(r)] * b[static_cast<size_t>(o * rank + r)];
        }
        out[static_cast<size_t>(o)] += scale * s;
    }
    return out;
}

struct CpuModuleRunner {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ggml = nullptr;
    engine::core::ModuleBuildContext ctx{};

    CpuModuleRunner() {
        backend = engine::core::init_backend(backend_config);
        ggml_init_params params{};
        params.mem_size = kTestGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to init test ggml context");
        }
        ctx.ggml = ggml;
        ctx.module_instance_name = "breeze_lora_side_adapter_test";
    }

    ~CpuModuleRunner() {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ggml != nullptr) ggml_free(ggml);
        if (backend != nullptr) ggml_backend_free(backend);
    }

    engine::core::TensorValue make_f32(std::initializer_list<int64_t> dims) {
        return engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(dims));
    }

    std::vector<float> run(const engine::core::TensorValue & a, const engine::core::TensorValue & b) {
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test backend tensors");
        }
        auto * graph = ggml_new_graph_custom(ggml, kTestGraphNodes, false);
        ggml_build_forward_expand(graph, a.tensor);
        ggml_build_forward_expand(graph, b.tensor);
        ggml_backend_graph_compute(backend, graph);
        std::vector<float> out;
        engine::core::read_tensor_f32_into(a.tensor, out);
        return out;
    }
};

}  // namespace

int main() {
    constexpr int64_t in_f = 4;
    constexpr int64_t out_f = 3;
    constexpr int64_t rank = kBreezeLoraRank;  // fixed Instavar layout (8)
    constexpr float scale = 2.0F;

    const std::vector<float> w = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<float> a_vals(static_cast<size_t>(rank * in_f), 0.0F);
    a_vals[0] = 1.0F;                     // row 0 selects input 0
    a_vals[static_cast<size_t>(in_f) + 1] = 1.0F;  // row 1 selects input 1
    std::vector<float> b_vals(static_cast<size_t>(out_f * rank), 0.0F);
    b_vals[0] = 1.0F;                     // out 0 <- rank 0
    b_vals[static_cast<size_t>(rank) + 1] = 1.0F;  // out 1 <- rank 1
    const std::vector<float> x = {1, 2, 3, 4};

    CpuModuleRunner runner;
    auto input = runner.make_f32({1, in_f});
    auto weight = runner.make_f32({out_f, in_f});
    auto a_t = runner.make_f32({rank, in_f});
    auto b_t = runner.make_f32({out_f, rank});

    BreezeLoraLinearContext adapter;
    adapter.a = LinearWeights{a_t, std::nullopt};
    adapter.b = LinearWeights{b_t, std::nullopt};
    adapter.scale = scale;
    LinearWeights base_w{weight, std::nullopt};

    auto enabled = build_adapter_linear(runner.ctx, input, base_w, in_f, out_f, GGML_PREC_DEFAULT, &adapter);
    auto disabled = build_adapter_linear(runner.ctx, input, base_w, in_f, out_f, GGML_PREC_DEFAULT, nullptr);
    ggml_set_output(enabled.tensor);
    ggml_set_output(disabled.tensor);

    const std::vector<float> x_row = x;
    const std::vector<float> w_row = w;
    const std::vector<float> a_row = a_vals;
    const std::vector<float> b_row = b_vals;

    // Allocate before uploading, mirroring the module-test pattern.
    runner.buffer = ggml_backend_alloc_ctx_tensors(runner.ggml, runner.backend);
    require(runner.buffer != nullptr, "buffer alloc");
    ggml_backend_tensor_set(input.tensor, x_row.data(), 0, x_row.size() * sizeof(float));
    ggml_backend_tensor_set(weight.tensor, w_row.data(), 0, w_row.size() * sizeof(float));
    ggml_backend_tensor_set(a_t.tensor, a_row.data(), 0, a_row.size() * sizeof(float));
    ggml_backend_tensor_set(b_t.tensor, b_row.data(), 0, b_row.size() * sizeof(float));

    auto * graph = ggml_new_graph_custom(runner.ggml, kTestGraphNodes, false);
    ggml_build_forward_expand(graph, enabled.tensor);
    ggml_build_forward_expand(graph, disabled.tensor);
    require(ggml_backend_graph_compute(runner.backend, graph) == GGML_STATUS_SUCCESS, "graph compute");

    std::vector<float> got_enabled;
    std::vector<float> got_disabled;
    engine::core::read_tensor_f32_into(enabled.tensor, got_enabled);
    engine::core::read_tensor_f32_into(disabled.tensor, got_disabled);

    const auto want_enabled = reference(w, a_vals, b_vals, scale, true, in_f, out_f, rank, x);
    const auto want_disabled = reference(w, a_vals, b_vals, scale, false, in_f, out_f, rank, x);

    require(got_enabled.size() == static_cast<size_t>(out_f), "enabled size");
    require(got_disabled.size() == static_cast<size_t>(out_f), "disabled size");
    for (int64_t i = 0; i < out_f; ++i) {
        require(
            nearly_equal(got_enabled[static_cast<size_t>(i)], want_enabled[static_cast<size_t>(i)]),
            "side adapter equals base + scale*B(A(x))");
        require(
            nearly_equal(got_disabled[static_cast<size_t>(i)], want_disabled[static_cast<size_t>(i)]),
            "null adapter equals base(x)");
    }
    bool changed = false;
    for (int64_t i = 0; i < out_f; ++i) {
        if (!nearly_equal(got_enabled[static_cast<size_t>(i)], got_disabled[static_cast<size_t>(i)])) {
            changed = true;
        }
    }
    require(changed, "active adapter changes the projection output");

    std::cout << "test_breeze_lora_side_adapter: ok\n";
    return 0;
}
