#include "config.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <unordered_set>

namespace breeze_lora_server {
namespace {

std::filesystem::path resolve_path(const std::filesystem::path & config_path, const std::string & value) {
    std::filesystem::path path(value);
    if (path.is_absolute()) {
        return path;
    }
    return (config_path.parent_path() / path).lexically_normal();
}

std::string load_reference_text(
    const std::filesystem::path & config_path,
    const engine::io::json::Value & item) {
    const auto * inline_text = item.find("reference_text");
    const auto * text_file = item.find("reference_text_file");
    const bool has_inline = inline_text != nullptr && !inline_text->is_null();
    const bool has_file = text_file != nullptr && !text_file->is_null();
    if (has_inline && has_file) {
        throw std::runtime_error(
            "model entry may set only one of reference_text or reference_text_file");
    }
    if (has_inline) {
        const auto text = inline_text->as_string();
        if (text.empty()) {
            throw std::runtime_error("reference_text must be non-empty when set");
        }
        return text;
    }
    if (has_file) {
        const auto path = resolve_path(config_path, text_file->as_string());
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("reference_text_file missing: " + path.string());
        }
        auto text = engine::io::read_text_file(path);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        if (text.empty()) {
            throw std::runtime_error("reference_text_file is empty: " + path.string());
        }
        return text;
    }
    return {};
}

}  // namespace

ServerConfig load_config(const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    ServerConfig config;
    config.config_dir = path.parent_path();
    config.host = engine::io::json::optional_string(root, "host", config.host);
    config.port = engine::io::json::optional_i32(root, "port", config.port);
    config.backend = engine::io::json::optional_string(root, "backend", config.backend);
    config.device = engine::io::json::optional_i32(root, "device", config.device);
    config.threads = engine::io::json::optional_i32(root, "threads", config.threads);
    config.max_queue_depth = engine::io::json::optional_i32(root, "max_queue_depth", config.max_queue_depth);
    config.base_revision = engine::io::json::optional_string(root, "base_revision", "");
    config.base_model = resolve_path(path, engine::io::json::require_string(root, "base_model"));
    if (config.port <= 0 || config.port > 65535) {
        throw std::runtime_error("port must be in 1..65535");
    }
    if (config.threads <= 0) {
        throw std::runtime_error("threads must be positive");
    }
    if (config.max_queue_depth <= 0) {
        throw std::runtime_error("max_queue_depth must be positive");
    }
    if (config.backend != "cuda" && config.backend != "cpu") {
        throw std::runtime_error("backend must be cuda or cpu");
    }
    if (!engine::io::is_existing_file(config.base_model) && !engine::io::is_existing_directory(config.base_model)) {
        throw std::runtime_error("base_model path does not exist: " + config.base_model.string());
    }

    const auto * models = root.find("models");
    if (models == nullptr || !models->is_array() || models->as_array().empty()) {
        throw std::runtime_error("models must be a non-empty array");
    }
    std::unordered_set<std::string> seen;
    bool has_base = false;
    for (const auto & item : models->as_array()) {
        if (!item.is_object()) {
            throw std::runtime_error("each models entry must be an object");
        }
        ModelEntry entry;
        entry.id = engine::io::json::require_string(item, "id");
        if (entry.id.empty()) {
            throw std::runtime_error("model id must be non-empty");
        }
        if (!seen.insert(entry.id).second) {
            throw std::runtime_error("duplicate model id: " + entry.id);
        }
        const auto * lora = item.find("lora");
        if (lora == nullptr || lora->is_null()) {
            if (entry.id != "breeze-base") {
                throw std::runtime_error("null lora entry must use id breeze-base");
            }
            has_base = true;
        } else {
            entry.lora = resolve_path(path, lora->as_string());
            if (!engine::io::is_existing_directory(*entry.lora)) {
                throw std::runtime_error("adapter directory missing: " + entry.lora->string());
            }
        }
        entry.default_instruction = engine::io::json::optional_string(
            item, "default_instruction", entry.default_instruction);
        if (entry.default_instruction.empty()) {
            throw std::runtime_error("default_instruction must be non-empty for model: " + entry.id);
        }

        const auto * voice_ref = item.find("voice_ref");
        const bool has_voice_ref = voice_ref != nullptr && !voice_ref->is_null();
        entry.reference_text = load_reference_text(path, item);
        if (has_voice_ref) {
            entry.voice_ref = resolve_path(path, voice_ref->as_string());
            if (!engine::io::is_existing_file(*entry.voice_ref)) {
                throw std::runtime_error("voice_ref missing: " + entry.voice_ref->string());
            }
            if (entry.reference_text.empty()) {
                throw std::runtime_error(
                    "model " + entry.id + " sets voice_ref but missing reference_text/reference_text_file");
            }
        } else if (!entry.reference_text.empty()) {
            throw std::runtime_error(
                "model " + entry.id + " sets reference text without voice_ref");
        }
        config.models.push_back(std::move(entry));
    }
    if (!has_base) {
        throw std::runtime_error("models must include breeze-base with lora: null");
    }
    return config;
}

}  // namespace breeze_lora_server
