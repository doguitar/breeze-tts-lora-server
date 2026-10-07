#include "config.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_set>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace breeze_lora_server {
namespace {

std::filesystem::path resolve_path(const std::filesystem::path & config_path, const std::string & value) {
    std::filesystem::path path(value);
    if (path.is_absolute()) {
        return path;
    }
    return (config_path.parent_path() / path).lexically_normal();
}

void validate_model_id(const std::string & id) {
    if (id.find('/') != std::string::npos || id.find('\\') != std::string::npos || id.find("..") != std::string::npos) {
        throw std::runtime_error("model id must not contain '/', '\\', or '..': " + id);
    }
}

std::string load_reference_text(
    const std::filesystem::path & config_path,
    const engine::io::json::Value & item,
    std::optional<std::string> & reference_text_file_source) {
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
        reference_text_file_source.reset();
        return text;
    }
    if (has_file) {
        const auto source = text_file->as_string();
        const auto path = resolve_path(config_path, source);
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
        reference_text_file_source = source;
        return text;
    }
    reference_text_file_source.reset();
    return {};
}

std::string json_quote(const std::string & value) {
    return engine::io::json::stringify_string(value);
}

}  // namespace

bool is_loopback_host(const std::string & host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

ServerConfig load_config(const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    ServerConfig config;
    config.config_path = std::filesystem::absolute(path).lexically_normal();
    config.config_dir = config.config_path.parent_path();
    config.host = engine::io::json::optional_string(root, "host", config.host);
    config.port = engine::io::json::optional_i32(root, "port", config.port);
    config.backend = engine::io::json::optional_string(root, "backend", config.backend);
    config.device = engine::io::json::optional_i32(root, "device", config.device);
    config.threads = engine::io::json::optional_i32(root, "threads", config.threads);
    config.max_queue_depth = engine::io::json::optional_i32(root, "max_queue_depth", config.max_queue_depth);
    config.base_revision = engine::io::json::optional_string(root, "base_revision", "");
    config.base_model_source = engine::io::json::require_string(root, "base_model");
    config.base_model = resolve_path(path, config.base_model_source);
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
        validate_model_id(entry.id);
        if (!seen.insert(entry.id).second) {
            throw std::runtime_error("duplicate model id: " + entry.id);
        }
        const auto * lora = item.find("lora");
        if (lora == nullptr || lora->is_null()) {
            if (entry.id != "breeze-base") {
                throw std::runtime_error("null lora entry must use id breeze-base");
            }
            has_base = true;
            entry.lora_source.clear();
        } else {
            entry.lora_source = lora->as_string();
            entry.lora = resolve_path(path, entry.lora_source);
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
        entry.reference_text = load_reference_text(path, item, entry.reference_text_file_source);
        if (has_voice_ref) {
            entry.voice_ref_source = voice_ref->as_string();
            entry.voice_ref = resolve_path(path, entry.voice_ref_source);
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

std::string serialize_config(const ServerConfig & config) {
    std::ostringstream out;
    out << "{\n"
        << "  \"host\": " << json_quote(config.host) << ",\n"
        << "  \"port\": " << config.port << ",\n"
        << "  \"backend\": " << json_quote(config.backend) << ",\n"
        << "  \"device\": " << config.device << ",\n"
        << "  \"threads\": " << config.threads << ",\n"
        << "  \"max_queue_depth\": " << config.max_queue_depth << ",\n"
        << "  \"base_model\": " << json_quote(config.base_model_source) << ",\n"
        << "  \"base_revision\": " << json_quote(config.base_revision) << ",\n"
        << "  \"models\": [\n";
    for (size_t i = 0; i < config.models.size(); ++i) {
        const auto & entry = config.models[i];
        out << "    {\n"
            << "      \"id\": " << json_quote(entry.id) << ",\n";
        if (!entry.lora.has_value()) {
            out << "      \"lora\": null,\n";
        } else {
            out << "      \"lora\": " << json_quote(entry.lora_source) << ",\n";
        }
        out << "      \"default_instruction\": " << json_quote(entry.default_instruction);
        if (entry.voice_ref.has_value()) {
            out << ",\n"
                << "      \"voice_ref\": " << json_quote(entry.voice_ref_source);
            if (entry.reference_text_file_source.has_value()) {
                out << ",\n"
                    << "      \"reference_text_file\": "
                    << json_quote(*entry.reference_text_file_source);
            } else {
                out << ",\n"
                    << "      \"reference_text\": " << json_quote(entry.reference_text);
            }
        }
        out << "\n    }";
        if (i + 1 < config.models.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n"
        << "}\n";
    return out.str();
}

void replace_file_atomically(
    const std::filesystem::path & source,
    const std::filesystem::path & destination) {
#if defined(_WIN32)
    // MoveFileEx can replace an existing destination without deleting it first.
    // std::filesystem::rename cannot on Windows, and delete-then-rename can lose
    // both the live file and the temp if the second rename fails.
    if (!::MoveFileExW(
            source.c_str(),
            destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD err = ::GetLastError();
        throw std::runtime_error(
            "failed to replace file: " + destination.string() + " (win32=" + std::to_string(err) + ")");
    }
#else
    std::error_code ec;
    std::filesystem::rename(source, destination, ec);
    if (ec) {
        throw std::runtime_error(
            "failed to replace file: " + destination.string() + ": " + ec.message());
    }
#endif
}

void save_config_atomically(const ServerConfig & config) {
    if (config.config_path.empty()) {
        throw std::runtime_error("cannot save config: config_path is empty");
    }
    const auto tmp_path = std::filesystem::path(config.config_path.string() + ".tmp");
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open temp config for write: " + tmp_path.string());
        }
        out << serialize_config(config);
        out.flush();
        if (!out) {
            out.close();
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            throw std::runtime_error("failed to write temp config: " + tmp_path.string());
        }
    }
    try {
        replace_file_atomically(tmp_path, config.config_path);
    } catch (...) {
        // Live destination is still intact; drop the temp publish attempt.
        std::error_code cleanup_ec;
        std::filesystem::remove(tmp_path, cleanup_ec);
        throw;
    }
}

}  // namespace breeze_lora_server
