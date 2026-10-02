// Regression: a PackedGateUp MLP with a side-adapter hook installed must build and
// run. The hook splits packed gate/up into two projections, so the packed tensor is
// no longer available for the single-tensor fused swiglu kernel. Before the fix the
// fused branch dereferenced the never-set packed tensor and crashed the process with
// SIGSEGV inside ggml_glu_impl during graph build.
//
// This exercises the real decoder MLP path, not a reimplementation of it.
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <ggml.h>

namespace {

namespace core = engine::core;
namespace modules = engine::modules;

constexpr size_t kGraphBytes = 16 * 1024 * 1024;
constexpr size_t kGraphNodes = 4096;

constexpr int64_t kHidden = 8;
constexpr int64_t kIntermediate = 12;
constexpr int64_t kSteps = 3;

std::vector<float> patterned(size_t count, float phase, float scale) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = scale * std::sin(phase + static_cast<float>(i) * 0.11F);
    }
    return out;
}

// Builds one decoder layer whose MLP is PackedGateUp, optionally with the
// side-adapter hook installed, and returns the layer output.
std::vector<float> run_layer_with_hook(bool install_hook) {
    core::BackendConfig backend_config{core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize test backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        core::ModuleBuildContext ctx{ggml, "decoder_packed_hook_test", core::BackendType::Cpu};
        auto make_f32 = [&](std::initializer_list<int64_t> dims) {
            return core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims(dims));
        };

        auto input = make_f32({1, kSteps, kHidden});
        auto positions = core::make_tensor(
            ctx, GGML_TYPE_I32, core::TensorShape::from_dims({kSteps}));

        modules::DecoderLayerWeights weights;
        weights.input_norm = {make_f32({kHidden}), std::nullopt};
        weights.post_norm = {make_f32({kHidden}), std::nullopt};
        weights.self_attention.out_weight = make_f32({kHidden, kHidden});
        weights.mlp.down_proj = {make_f32({kHidden, kIntermediate}), std::nullopt};

        constexpr int64_t heads = 2;
        constexpr int64_t kv_heads = 1;
        constexpr int64_t head_dim = 4;
        constexpr int64_t q_out = heads * head_dim;
        constexpr int64_t kv_out = kv_heads * head_dim;

        weights.self_attention.qkv_weight = make_f32({q_out + 2 * kv_out, kHidden});
        // Packed gate/up weight the hook will split into separate projections.
        weights.mlp.gate_up_proj = modules::LinearWeights{make_f32({kIntermediate * 2, kHidden}), std::nullopt};

        modules::DecoderLayerConfig config;
        config.hidden_size = kHidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = kIntermediate;
        config.rms_norm_eps = 1e-5F;
        config.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
        config.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
        config.use_qk_norm = false;
        config.runtime.attention.prefill_mode = modules::DecoderAttentionMode::ManualRepeat;

        if (install_hook) {
            // Exactly the shape the Breeze generator installs. A null adapter context
            // keeps the base projection, which is what the server does for breeze-base.
            config.side_adapter_linear = [](
                                             core::ModuleBuildContext & build,
                                             const std::string &,
                                             const core::TensorValue & hook_input,
                                             const modules::LinearWeights & hook_weights,
                                             int64_t in_features,
                                             int64_t out_features,
                                             ggml_prec precision) {
                return modules::LinearModule({in_features, out_features, hook_weights.bias.has_value(), precision})
                    .build(build, hook_input, hook_weights);
            };
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        const auto outputs = modules::DecoderLayerModule(config).build(ctx, input, positions, weights);
        ggml_build_forward_expand(graph, outputs.output.tensor);

        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test tensors");
        }

        core::write_tensor_f32(input, patterned(static_cast<size_t>(kSteps * kHidden), 2.1F, 0.20F));
        core::write_tensor_i32(positions, std::vector<int32_t>{0, 1, 2});
        core::write_tensor_f32(*weights.input_norm.weight, patterned(kHidden, 0.3F, 0.7F));
        core::write_tensor_f32(*weights.post_norm.weight, patterned(kHidden, 0.7F, 0.8F));
        core::write_tensor_f32(
            weights.self_attention.out_weight, patterned(static_cast<size_t>(kHidden * kHidden), 1.1F, 0.10F));
        core::write_tensor_f32(
            weights.mlp.down_proj.weight,
            patterned(static_cast<size_t>(kHidden * kIntermediate), 1.5F, 0.10F));
        core::write_tensor_f32(
            *weights.self_attention.qkv_weight,
            patterned(static_cast<size_t>((q_out + 2 * kv_out) * kHidden), 0.1F, 0.12F));

        // Packed gate/up rows: first half gate, second half up.
        const auto gate_values = patterned(static_cast<size_t>(kIntermediate * kHidden), 1.3F, 0.11F);
        const auto up_values = patterned(static_cast<size_t>(kIntermediate * kHidden), 1.7F, 0.09F);
        std::vector<float> packed;
        packed.insert(packed.end(), gate_values.begin(), gate_values.end());
        packed.insert(packed.end(), up_values.begin(), up_values.end());
        core::write_tensor_f32(weights.mlp.gate_up_proj->weight, packed);

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }

        std::vector<float> out;
        core::read_tensor_f32_into(outputs.output.tensor, out);

        ggml_backend_buffer_free(buffer);
        ggml_free(ggml);
        ggml_backend_free(backend);
        return out;
    } catch (...) {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

}  // namespace

int main() {
    // No hook: the single-tensor fused swiglu path.
    const auto plain = run_layer_with_hook(false);
    engine::test::require(plain.size() == static_cast<size_t>(kSteps * kHidden), "plain output size");

    // Hook installed: this is the case that used to crash during graph build.
    const auto hooked = run_layer_with_hook(true);
    engine::test::require(hooked.size() == static_cast<size_t>(kSteps * kHidden), "hooked output size");

    // A null-adapter hook must reproduce the unadapted projection exactly.
    for (size_t i = 0; i < plain.size(); ++i) {
        engine::test::require(
            std::isfinite(hooked[i]),
            "hooked output must be finite");
        engine::test::require(
            std::fabs(plain[i] - hooked[i]) < 1e-5F,
            "hook with no adapter must equal the packed path");
    }

    std::cout << "test_decoder_packed_hook: ok\n";
    return 0;
}
