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
    std::string default_instruction = "Speak clearly and naturally.";
    // Optional clone defaults. If either is set, both must resolve.
    std::optional<std::filesystem::path> voice_ref;
    std::string reference_text;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string backend = "cuda";
    int device = 0;
    int threads = 1;
    int max_queue_depth = 8;
    std::filesystem::path base_model;
    std::string base_revision;
    std::filesystem::path config_dir;  // parent of the loaded server.json
    std::vector<ModelEntry> models;
};

ServerConfig load_config(const std::filesystem::path & path);

}  // namespace breeze_lora_server
