#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace breeze_lora_server {

struct ModelEntry {
    std::string id;
    std::optional<std::filesystem::path> lora;  // nullopt => breeze-base
    // Original JSON spelling for lora when set; empty when lora is null.
    std::string lora_source;
    std::string default_instruction = "Speak clearly and naturally.";
    // Optional clone defaults. If either is set, both must resolve.
    std::optional<std::filesystem::path> voice_ref;
    // Original JSON spelling for voice_ref when set.
    std::string voice_ref_source;
    std::string reference_text;
    // When set, serialize as reference_text_file with this spelling; otherwise
    // emit inline reference_text when non-empty.
    std::optional<std::string> reference_text_file_source;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string backend = "cuda";
    int device = 0;
    int threads = 1;
    int max_queue_depth = 8;
    std::filesystem::path base_model;
    std::string base_model_source;
    std::string base_revision;
    std::filesystem::path config_path;  // absolute path of loaded server.json
    std::filesystem::path config_dir;   // parent of the loaded server.json
    std::vector<ModelEntry> models;
};

ServerConfig load_config(const std::filesystem::path & path);
std::string serialize_config(const ServerConfig & config);
void save_config_atomically(const ServerConfig & config);
bool is_loopback_host(const std::string & host);

}  // namespace breeze_lora_server
