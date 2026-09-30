#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
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
    void bind_target(BreezeLoraTargetBinding binding);
    void register_adapter(const std::string & model_id, BreezeLoraAdapterTensors adapter);
    void set_base_identity(std::string revision, std::unordered_map<std::string, std::string> file_hashes);

    const std::string & active_id() const noexcept { return active_id_; }
    size_t activation_count() const noexcept { return activation_count_; }
    const std::vector<std::string> & model_ids() const noexcept { return model_ids_; }

    void activate_base();
    void activate_adapter(const std::string & model_id);

private:
    struct LiveBuffer {
        ggml_tensor * live = nullptr;
        engine::core::TensorShape shape = {};
        ggml_type type = GGML_TYPE_F32;
        std::vector<float> staging_f32;
        std::string debug_name;
    };

    void ensure_model_id_list();
    void upload_live_buffers();

    std::vector<BreezeLoraTargetBinding> targets_;
    std::unordered_map<std::string, BreezeLoraAdapterTensors> adapters_;
    std::vector<std::string> model_ids_;
    std::string active_id_ = kBreezeBaseModelId;
    size_t activation_count_ = 0;
    std::string configured_revision_;
    std::unordered_map<std::string, std::string> configured_file_hashes_;
    std::unordered_map<ggml_tensor *, LiveBuffer> live_buffers_;
};

}  // namespace engine::models::breeze_tts
