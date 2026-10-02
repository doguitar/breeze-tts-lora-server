#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace engine::models::breeze_tts {

inline constexpr const char * kBreezeLoraVariant = "backbone_depth_projection";
inline constexpr int kBreezeLoraRank = 8;
inline constexpr float kBreezeLoraAlpha = 16.0F;
inline constexpr const char * kBreezeBaseModelId = "breeze-base";

struct BreezeLoraManifest {
    std::string base_model_id;
    std::string base_revision;
    std::unordered_map<std::string, std::string> base_files;
    std::string adapter_file = "adapter.safetensors";
    std::string adapter_sha256;
    std::string variant = kBreezeLoraVariant;
    int rank = kBreezeLoraRank;
    float alpha = kBreezeLoraAlpha;
    int seed = 0;
};

struct BreezeLoraAdapterTensors {
    std::unordered_map<std::string, std::vector<float>> lora_a;
    std::unordered_map<std::string, std::vector<float>> lora_b;
    float scale = kBreezeLoraAlpha / static_cast<float>(kBreezeLoraRank);
    BreezeLoraManifest manifest;
};

struct BreezeLoraTargetBinding {
    std::string module_name;
    int64_t out_features = 0;
    int64_t in_features = 0;
    std::vector<float> base_f32;
    ggml_tensor * live = nullptr;
    engine::core::TensorShape live_shape = {};
    ggml_type live_type = GGML_TYPE_F32;
    int64_t row_offset = 0;
};

void breeze_lora_linear_forward(
    const std::vector<float> & weight,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    float scale,
    bool enabled,
    int64_t out_features,
    int64_t in_features,
    int64_t rank,
    const std::vector<float> & input,
    int64_t batch,
    std::vector<float> & output);

std::vector<float> breeze_lora_merge_weight(
    const std::vector<float> & weight,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    float scale,
    int64_t out_features,
    int64_t in_features,
    int64_t rank);

std::string sha256_hex_file(const std::filesystem::path & path);
std::string sha256_hex_bytes(const void * data, size_t size);

BreezeLoraManifest load_breeze_lora_manifest(const std::filesystem::path & adapter_root);
void validate_breeze_lora_manifest(const BreezeLoraManifest & manifest);
BreezeLoraAdapterTensors load_breeze_lora_adapter(
    const std::filesystem::path & adapter_root,
    const engine::assets::TensorSource & base_weights);

class BreezeLoraManager {
public:
    struct SideBranch {
        modules::LinearWeights a;
        modules::LinearWeights b;
        float scale = 1.0F;
    };

    void bind_target(BreezeLoraTargetBinding binding);
    void register_adapter(const std::string & model_id, BreezeLoraAdapterTensors adapter);
    void set_base_identity(std::string revision, std::unordered_map<std::string, std::string> file_hashes);

    const std::string & active_id() const noexcept { return active_id_; }
    size_t activation_count() const noexcept { return activation_count_; }
    const std::vector<std::string> & model_ids() const noexcept { return model_ids_; }

    void activate_base();
    void activate_adapter(const std::string & model_id);

    // Uploads the pristine base rows into the live ggml tensors once, at startup.
    void upload_base_live_buffers();

    // Base rows are immutable; activation only swaps the active A/B/scale side adapter.
    const std::vector<float> & base_rows(ggml_tensor * live) const;
    const std::string & active_adapter_id() const noexcept { return active_adapter_id_; }
    const std::vector<float> & active_lora_a(const std::string & module_name) const;
    const std::vector<float> & active_lora_b(const std::string & module_name) const;
    const BreezeLoraAdapterTensors * active_adapter() const noexcept { return active_adapter_; }

    // Executable side branches: one A/B ggml tensor pair per adapter module,
    // created inside the Breeze weight store and addressed by name. `modules`
    // nullopt allocates every module; otherwise only the listed ones.
    void create_side_branch_tensors(
        const std::string & model_id,
        engine::core::BackendWeightStore & store,
        const std::unordered_set<std::string> * modules = nullptr);
    const SideBranch * side_branch(const std::string & model_id, const std::string & module_name) const;
    const SideBranch * active_side_branch(const std::string & module_name) const;
    const std::vector<BreezeLoraTargetBinding> & targets() const noexcept { return targets_; }
    // Re-checks every registered adapter against the given bound targets.
    void validate_registered_adapters(const std::unordered_set<std::string> * modules = nullptr) const;

private:
    struct LiveBuffer {
        ggml_tensor * live = nullptr;
        engine::core::TensorShape shape = {};
        ggml_type type = GGML_TYPE_F32;
        std::vector<float> base_f32;  // pristine unadapted base rows; written once at bind
        std::string debug_name;
    };

    void ensure_model_id_list();

    std::vector<BreezeLoraTargetBinding> targets_;
    std::unordered_map<std::string, BreezeLoraAdapterTensors> adapters_;
    std::unordered_map<std::string, std::unordered_map<std::string, SideBranch>> side_branches_;
    std::vector<std::string> model_ids_;
    std::string active_id_ = kBreezeBaseModelId;
    std::string active_adapter_id_;
    size_t activation_count_ = 0;
    const BreezeLoraAdapterTensors * active_adapter_ = nullptr;
    std::string configured_revision_;
    std::unordered_map<std::string, std::string> configured_file_hashes_;
    std::unordered_map<ggml_tensor *, LiveBuffer> live_buffers_;
};

}  // namespace engine::models::breeze_tts
