#include "engine/models/breeze_tts/lora_linear.h"

#include <stdexcept>

namespace engine::models::breeze_tts {

core::TensorValue build_adapter_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features,
    ggml_prec precision,
    const BreezeLoraLinearContext * adapter) {
    auto base = modules::LinearModule({in_features, out_features, weights.bias.has_value(), precision})
                    .build(ctx, input, weights);
    if (adapter == nullptr) {
        return base;
    }
    if (in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("Breeze LoRA side adapter requires positive linear features");
    }
    if (!(adapter->scale > 0.0F)) {
        throw std::runtime_error("Breeze LoRA side adapter scale must be positive");
    }
    // side = alpha/rank * (x @ A^T) @ B^T, accumulated into the frozen base output.
    auto inner = modules::LinearModule({in_features, kBreezeLoraRank, false, precision})
                     .build(ctx, input, adapter->a);
    auto outer = modules::LinearModule({kBreezeLoraRank, out_features, false, precision})
                     .build(ctx, inner, adapter->b);
    auto scaled = core::wrap_tensor(
        ggml_scale(ctx.ggml, outer.tensor, adapter->scale),
        outer.shape,
        outer.type);
    return core::wrap_tensor(
        ggml_add(ctx.ggml, base.tensor, scaled.tensor),
        base.shape,
        base.type);
}

BreezeLoraLinearScope * BreezeLoraLinearScope::active_ = nullptr;

BreezeLoraLinearScope::BreezeLoraLinearScope(BreezeLoraContextLookup lookup, void * user_data)
    : previous_(active_), lookup_(lookup), user_data_(user_data) {
    active_ = this;
}

BreezeLoraLinearScope::~BreezeLoraLinearScope() {
    active_ = previous_;
}

const BreezeLoraLinearContext * BreezeLoraLinearScope::resolve(const std::string & module_name) {
    if (active_ == nullptr || active_->lookup_ == nullptr) {
        return nullptr;
    }
    return active_->lookup_(active_->user_data_, module_name);
}

}  // namespace engine::models::breeze_tts
