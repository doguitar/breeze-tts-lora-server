#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/breeze_tts/lora.h"

#include <cstdint>
#include <string>

namespace engine::models::breeze_tts {

// Optional side-adapter context for Breeze decoder graphs.
//
// When attached, `build_adapter_linear` computes `base(x) + scale * B(A(x))`
// from tensors loaded in the side adapter's own ggml context; the base weight,
// bias, and graph tensors are never mutated. A null context keeps the exact
// `LinearModule` behavior.
struct BreezeLoraLinearContext {
    modules::LinearWeights a;
    modules::LinearWeights b;
    float scale = 1.0F;
};

// Resolves the side-adapter context for one module name. `nullptr` means the
// active model is `breeze-base` or the module is not adapted.
using BreezeLoraLinearResolver = const BreezeLoraLinearContext * (*)(
    void * user_data,
    const std::string & module_name);

// Builds a linear projection with an optional Instavar side adapter.
core::TensorValue build_adapter_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features,
    ggml_prec precision,
    const BreezeLoraLinearContext * adapter);

// Adapter lookup performed at graph build time, in the owning model's namespace.
using BreezeLoraContextLookup = BreezeLoraLinearContext * (*)(
    void * user_data,
    const std::string & module_name);

class BreezeLoraLinearScope {
public:
    BreezeLoraLinearScope(BreezeLoraContextLookup lookup, void * user_data);
    ~BreezeLoraLinearScope();
    BreezeLoraLinearScope(const BreezeLoraLinearScope &) = delete;
    BreezeLoraLinearScope & operator=(const BreezeLoraLinearScope &) = delete;

    static const BreezeLoraLinearContext * resolve(const std::string & module_name);

private:
    static BreezeLoraLinearScope * active_;
    BreezeLoraLinearScope * previous_ = nullptr;
    BreezeLoraContextLookup lookup_ = nullptr;
    void * user_data_ = nullptr;
};

}  // namespace engine::models::breeze_tts
