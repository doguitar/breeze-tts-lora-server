#include "engine/models/breeze_tts/lora.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <ggml-backend.h>

#include <algorithm>
#include <string>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string_view>
#include <stdexcept>

namespace engine::models::breeze_tts {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;
namespace io = engine::io;


constexpr std::string_view kManifestName = "adapter_config.json";

const std::array<const char *, 7> kTransformerSuffixes = {
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
};

const std::array<const char *, 3> kProjectionModules = {
    "text_encoder_proj",
    "depth_decoder.model.inputs_embeds_projector",
    "lm_head",
};

uint32_t rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

std::string to_hex(const std::array<uint8_t, 32> & digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        out[i * 2] = kHex[(digest[i] >> 4) & 0xF];
        out[i * 2 + 1] = kHex[digest[i] & 0xF];
    }
    return out;
}

std::array<uint8_t, 32> sha256(const uint8_t * data, size_t length) {
    static constexpr uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t, 8> h = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<uint8_t> msg(data, data + length);
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0);
    const uint64_t bit_len = static_cast<uint64_t>(length) * 8ULL;
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFF));

    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(msg[offset + i * 4]) << 24) | (uint32_t(msg[offset + i * 4 + 1]) << 16) |
                   (uint32_t(msg[offset + i * 4 + 2]) << 8) | uint32_t(msg[offset + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + maj;
            hh=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<uint8_t>((h[i] >> 24) & 0xFF);
        out[i * 4 + 1] = static_cast<uint8_t>((h[i] >> 16) & 0xFF);
        out[i * 4 + 2] = static_cast<uint8_t>((h[i] >> 8) & 0xFF);
        out[i * 4 + 3] = static_cast<uint8_t>(h[i] & 0xFF);
    }
    return out;
}

bool ends_with(const std::string & value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_lora_target_module(const std::string & module_name) {
    if (module_name.rfind("backbone_model.layers.", 0) == 0) {
        for (const char * suffix : kTransformerSuffixes) {
            if (ends_with(module_name, suffix)) return true;
        }
    }
    if (module_name.rfind("depth_decoder.model.layers.", 0) == 0) {
        for (const char * suffix : kTransformerSuffixes) {
            if (ends_with(module_name, suffix)) return true;
        }
    }
    for (const char * name : kProjectionModules) {
        if (module_name == name) return true;
    }
    return false;
}

void copy_rows(
    std::vector<float> & dst,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t row_offset,
    const std::vector<float> & src,
    int64_t src_rows,
    int64_t src_cols) {
    if (dst_cols != src_cols) {
        throw std::runtime_error("LoRA packed row copy column mismatch");
    }
    if (row_offset < 0 || src_rows < 0 || row_offset + src_rows > dst_rows) {
        throw std::runtime_error("LoRA packed row copy out of range");
    }
    for (int64_t row = 0; row < src_rows; ++row) {
        const size_t src_begin = static_cast<size_t>(row * src_cols);
        const size_t dst_begin = static_cast<size_t>((row_offset + row) * dst_cols);
        std::copy(
            src.begin() + static_cast<std::ptrdiff_t>(src_begin),
            src.begin() + static_cast<std::ptrdiff_t>(src_begin + src_cols),
            dst.begin() + static_cast<std::ptrdiff_t>(dst_begin));
    }
}

}  // namespace

std::string sha256_hex_bytes(const void * data, size_t size) {
    const auto digest = sha256(static_cast<const uint8_t *>(data), size);
    return to_hex(digest);
}

std::string sha256_hex_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for hashing: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const auto bytes = buffer.str();
    return sha256_hex_bytes(bytes.data(), bytes.size());
}

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
    std::vector<float> & output) {
    if (out_features <= 0 || in_features <= 0 || rank <= 0 || batch <= 0) {
        throw std::runtime_error("invalid LoRA linear shapes");
    }
    if (static_cast<int64_t>(weight.size()) != out_features * in_features ||
        static_cast<int64_t>(lora_a.size()) != rank * in_features ||
        static_cast<int64_t>(lora_b.size()) != out_features * rank ||
        static_cast<int64_t>(input.size()) != batch * in_features) {
        throw std::runtime_error("LoRA linear tensor size mismatch");
    }
    output.assign(static_cast<size_t>(batch * out_features), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            float sum = 0.0F;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += input[static_cast<size_t>(b * in_features + i)] *
                       weight[static_cast<size_t>(o * in_features + i)];
            }
            output[static_cast<size_t>(b * out_features + o)] = sum;
        }
    }
    if (!enabled) {
        return;
    }
    std::vector<float> mid(static_cast<size_t>(batch * rank), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rank; ++r) {
            float sum = 0.0F;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += input[static_cast<size_t>(b * in_features + i)] *
                       lora_a[static_cast<size_t>(r * in_features + i)];
            }
            mid[static_cast<size_t>(b * rank + r)] = sum;
        }
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            float sum = 0.0F;
            for (int64_t r = 0; r < rank; ++r) {
                sum += mid[static_cast<size_t>(b * rank + r)] *
                       lora_b[static_cast<size_t>(o * rank + r)];
            }
            output[static_cast<size_t>(b * out_features + o)] += sum * scale;
        }
    }
}

std::vector<float> breeze_lora_merge_weight(
    const std::vector<float> & weight,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    float scale,
    int64_t out_features,
    int64_t in_features,
    int64_t rank) {
    if (out_features <= 0 || in_features <= 0 || rank <= 0) {
        throw std::runtime_error("invalid LoRA merge shapes");
    }
    if (static_cast<int64_t>(weight.size()) != out_features * in_features ||
        static_cast<int64_t>(lora_a.size()) != rank * in_features ||
        static_cast<int64_t>(lora_b.size()) != out_features * rank) {
        throw std::runtime_error("LoRA merge tensor size mismatch");
    }
    if (!std::isfinite(scale)) {
        throw std::runtime_error("LoRA scale must be finite");
    }
    std::vector<float> merged = weight;
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t i = 0; i < in_features; ++i) {
            float delta = 0.0F;
            for (int64_t r = 0; r < rank; ++r) {
                delta += lora_b[static_cast<size_t>(o * rank + r)] *
                         lora_a[static_cast<size_t>(r * in_features + i)];
            }
            merged[static_cast<size_t>(o * in_features + i)] += scale * delta;
        }
    }
    return merged;
}

void validate_breeze_lora_manifest(const BreezeLoraManifest & manifest) {
    if (manifest.variant != kBreezeLoraVariant) {
        throw std::runtime_error(
            "unsupported LoRA variant '" + manifest.variant + "'; expected " + kBreezeLoraVariant);
    }
    if (manifest.rank != kBreezeLoraRank) {
        throw std::runtime_error(
            "unsupported LoRA rank " + std::to_string(manifest.rank) + "; expected " +
            std::to_string(kBreezeLoraRank));
    }
    if (!(manifest.alpha > 0.0F) || !std::isfinite(manifest.alpha)) {
        throw std::runtime_error("LoRA alpha must be a positive finite value");
    }
    if (manifest.adapter_file.empty() ||
        std::filesystem::path(manifest.adapter_file).filename() != manifest.adapter_file) {
        throw std::runtime_error("adapter file must be a top-level filename");
    }
    if (manifest.adapter_sha256.size() != 64) {
        throw std::runtime_error("adapter sha256 must be 64 hex characters");
    }
    if (manifest.base_revision.empty()) {
        throw std::runtime_error("adapter manifest is missing base revision");
    }
    if (manifest.base_files.empty()) {
        throw std::runtime_error("adapter manifest base_model.files must be non-empty");
    }
}

BreezeLoraManifest load_breeze_lora_manifest(const std::filesystem::path & adapter_root) {
    const auto path = adapter_root / kManifestName;
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("adapter_config.json is missing: " + path.string());
    }
    const auto root = engine::io::json::parse_file(path);
    if (engine::io::json::optional_i64(root, "schema_version", 0) != 1) {
        throw std::runtime_error("unsupported adapter manifest schema");
    }
    if (engine::io::json::optional_string(root, "artifact_type", "") != "breeze_lora_adapter") {
        throw std::runtime_error("manifest is not a Breeze LoRA adapter");
    }
    const auto * base = root.find("base_model");
    const auto * adapter = root.find("adapter");
    const auto * lora = root.find("lora");
    if (base == nullptr || adapter == nullptr || lora == nullptr ||
        !base->is_object() || !adapter->is_object() || !lora->is_object()) {
        throw std::runtime_error("adapter manifest sections are incomplete");
    }
    BreezeLoraManifest manifest;
    manifest.base_model_id = engine::io::json::require_string(*base, "id");
    manifest.base_revision = engine::io::json::require_string(*base, "revision");
    const auto * files = base->find("files");
    if (files == nullptr || !files->is_object()) {
        throw std::runtime_error("adapter manifest base_model.files must be an object");
    }
    for (const auto & [name, value] : files->as_object()) {
        if (!value.is_string()) {
            throw std::runtime_error("invalid base model checksum for " + name);
        }
        const auto hash = value.as_string();
        if (hash.size() != 64) {
            throw std::runtime_error("invalid base model checksum for " + name);
        }
        const auto relative = std::filesystem::path(name);
        if (relative.is_absolute() || name.find("..") != std::string::npos) {
            throw std::runtime_error("unsafe base model path: " + name);
        }
        manifest.base_files.emplace(name, hash);
    }
    manifest.adapter_file = engine::io::json::require_string(*adapter, "file");
    manifest.adapter_sha256 = engine::io::json::require_string(*adapter, "sha256");
    manifest.variant = engine::io::json::require_string(*lora, "variant");
    manifest.rank = static_cast<int>(engine::io::json::require_i64(*lora, "rank"));
    manifest.alpha = engine::io::json::require_f32(*lora, "alpha");
    manifest.seed = static_cast<int>(engine::io::json::optional_i64(*lora, "seed", 0));
    validate_breeze_lora_manifest(manifest);
    return manifest;
}

BreezeLoraAdapterTensors load_breeze_lora_adapter(
    const std::filesystem::path & adapter_root,
    const engine::assets::TensorSource & base_weights) {
    auto manifest = load_breeze_lora_manifest(adapter_root);
    const auto adapter_path = adapter_root / manifest.adapter_file;
    if (!engine::io::is_existing_file(adapter_path)) {
        throw std::runtime_error("adapter file is missing: " + adapter_path.string());
    }
    const auto observed = sha256_hex_file(adapter_path);
    if (observed != manifest.adapter_sha256) {
        throw std::runtime_error("adapter file checksum mismatch");
    }
    auto adapter_source = engine::assets::open_tensor_source(adapter_path);
    BreezeLoraAdapterTensors out;
    out.manifest = std::move(manifest);
    out.scale = out.manifest.alpha / static_cast<float>(out.manifest.rank);

    std::unordered_map<std::string, bool> seen;
    for (const auto & metadata : adapter_source->tensors()) {
        const std::string & name = metadata.name;
        std::string module;
        bool is_a = false;
        if (ends_with(name, ".lora_A")) {
            module = name.substr(0, name.size() - 7);
            is_a = true;
        } else if (ends_with(name, ".lora_B")) {
            module = name.substr(0, name.size() - 7);
            is_a = false;
        } else {
            throw std::runtime_error("unexpected adapter tensor: " + name);
        }
        if (!is_lora_target_module(module)) {
            throw std::runtime_error("adapter targets unsupported module: " + module);
        }
        const auto base_name = module + ".weight";
        if (!base_weights.has_tensor(base_name)) {
            throw std::runtime_error("adapter module missing from base weights: " + base_name);
        }
        const auto base_meta = base_weights.require_metadata(base_name);
        if (base_meta.shape.size() != 2) {
            throw std::runtime_error("base weight must be rank-2: " + base_name);
        }
        const int64_t out_features = base_meta.shape[0];
        const int64_t in_features = base_meta.shape[1];
        if (is_a) {
            auto values = adapter_source->require_f32(name, {out.manifest.rank, in_features});
            out.lora_a.emplace(module, std::move(values));
        } else {
            auto values = adapter_source->require_f32(name, {out_features, out.manifest.rank});
            out.lora_b.emplace(module, std::move(values));
        }
        seen[module] = true;
    }
    for (const auto & [module, _] : seen) {
        if (!out.lora_a.count(module) || !out.lora_b.count(module)) {
            throw std::runtime_error("adapter is missing an A/B pair for " + module);
        }
    }
    if (seen.empty()) {
        throw std::runtime_error("adapter contains no LoRA tensors");
    }
    return out;
}

void BreezeLoraManager::bind_target(BreezeLoraTargetBinding binding) {
    if (binding.live == nullptr) {
        throw std::runtime_error("LoRA target live tensor is null: " + binding.module_name);
    }
    if (binding.out_features <= 0 || binding.in_features <= 0) {
        throw std::runtime_error("invalid LoRA target shape: " + binding.module_name);
    }
    if (static_cast<int64_t>(binding.base_f32.size()) != binding.out_features * binding.in_features) {
        throw std::runtime_error("LoRA target base size mismatch: " + binding.module_name);
    }
    if (binding.live_shape.rank < 2) {
        throw std::runtime_error("LoRA live tensor must be at least rank-2: " + binding.module_name);
    }
    auto & live = live_buffers_[binding.live];
    if (live.live == nullptr) {
        live.live = binding.live;
        live.shape = binding.live_shape;
        live.type = binding.live_type;
        live.staging_f32.assign(static_cast<size_t>(live.shape.num_elements()), 0.0F);
        live.debug_name = binding.module_name;
    } else if (live.shape.rank != binding.live_shape.rank ||
               live.shape.num_elements() != binding.live_shape.num_elements() ||
               live.type != binding.live_type) {
        throw std::runtime_error("conflicting LoRA live tensor metadata for " + binding.module_name);
    }
    targets_.push_back(std::move(binding));
}

void BreezeLoraManager::set_base_identity(
    std::string revision,
    std::unordered_map<std::string, std::string> file_hashes) {
    configured_revision_ = std::move(revision);
    configured_file_hashes_ = std::move(file_hashes);
}

void BreezeLoraManager::ensure_model_id_list() {
    model_ids_.clear();
    model_ids_.push_back(kBreezeBaseModelId);
    std::vector<std::string> ids;
    ids.reserve(adapters_.size());
    for (const auto & [id, _] : adapters_) {
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    model_ids_.insert(model_ids_.end(), ids.begin(), ids.end());
}

void BreezeLoraManager::register_adapter(const std::string & model_id, BreezeLoraAdapterTensors adapter) {
    if (model_id.empty() || model_id == kBreezeBaseModelId) {
        throw std::runtime_error("invalid adapter model id");
    }
    if (adapters_.count(model_id)) {
        throw std::runtime_error("duplicate adapter model id: " + model_id);
    }
    validate_breeze_lora_manifest(adapter.manifest);
    if (!configured_revision_.empty() && adapter.manifest.base_revision != configured_revision_) {
        throw std::runtime_error(
            "base revision mismatch for " + model_id + ": expected " + configured_revision_ +
            ", got " + adapter.manifest.base_revision);
    }
    for (const auto & [name, expected] : adapter.manifest.base_files) {
        const auto it = configured_file_hashes_.find(name);
        if (it != configured_file_hashes_.end() && it->second != expected) {
            throw std::runtime_error("base model checksum mismatch for " + name);
        }
    }
    for (const auto & target : targets_) {
        if (!adapter.lora_a.count(target.module_name) || !adapter.lora_b.count(target.module_name)) {
            throw std::runtime_error(
                "adapter " + model_id + " is missing required module " + target.module_name);
        }
        const auto & a = adapter.lora_a.at(target.module_name);
        const auto & b = adapter.lora_b.at(target.module_name);
        if (static_cast<int64_t>(a.size()) != kBreezeLoraRank * target.in_features ||
            static_cast<int64_t>(b.size()) != target.out_features * kBreezeLoraRank) {
            throw std::runtime_error("adapter shape mismatch for " + target.module_name);
        }
    }
    adapters_.emplace(model_id, std::move(adapter));
    ensure_model_id_list();
}

void BreezeLoraManager::upload_live_buffers() {
    for (auto & [tensor, buffer] : live_buffers_) {
        engine::assets::set_backend_tensor_from_f32_parallel(
            buffer.live,
            buffer.debug_name,
            buffer.staging_f32,
            buffer.shape,
            buffer.type,
            nullptr);
    }
}

void BreezeLoraManager::activate_base() {
    if (active_id_ == kBreezeBaseModelId) {
        return;
    }
    for (auto & [tensor, buffer] : live_buffers_) {
        std::fill(buffer.staging_f32.begin(), buffer.staging_f32.end(), 0.0F);
    }
    for (const auto & target : targets_) {
        auto & buffer = live_buffers_.at(target.live);
        const int64_t live_rows = buffer.shape.at(0);
        const int64_t live_cols = buffer.shape.at(1);
        copy_rows(
            buffer.staging_f32,
            live_rows,
            live_cols,
            target.row_offset,
            target.base_f32,
            target.out_features,
            target.in_features);
    }
    upload_live_buffers();
    active_id_ = kBreezeBaseModelId;
    ++activation_count_;
    engine::debug::log_message(engine::debug::LogLevel::Info, "breeze_tts.lora", "activated breeze-base");
}

void BreezeLoraManager::activate_adapter(const std::string & model_id) {
    if (model_id == kBreezeBaseModelId) {
        activate_base();
        return;
    }
    if (active_id_ == model_id) {
        return;
    }
    const auto it = adapters_.find(model_id);
    if (it == adapters_.end()) {
        throw std::runtime_error("unknown LoRA model id: " + model_id);
    }
    const auto & adapter = it->second;
    for (auto & [tensor, buffer] : live_buffers_) {
        std::fill(buffer.staging_f32.begin(), buffer.staging_f32.end(), 0.0F);
    }
    for (const auto & target : targets_) {
        const auto merged = breeze_lora_merge_weight(
            target.base_f32,
            adapter.lora_a.at(target.module_name),
            adapter.lora_b.at(target.module_name),
            adapter.scale,
            target.out_features,
            target.in_features,
            kBreezeLoraRank);
        auto & buffer = live_buffers_.at(target.live);
        const int64_t live_rows = buffer.shape.at(0);
        const int64_t live_cols = buffer.shape.at(1);
        copy_rows(
            buffer.staging_f32,
            live_rows,
            live_cols,
            target.row_offset,
            merged,
            target.out_features,
            target.in_features);
    }
    upload_live_buffers();
    active_id_ = model_id;
    ++activation_count_;
    engine::debug::log_message(
        engine::debug::LogLevel::Info,
        "breeze_tts.lora",
        "activated adapter " + model_id);
}

}  // namespace engine::models::breeze_tts
