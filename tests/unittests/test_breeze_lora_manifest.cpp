#include "engine/framework/io/safetensors.h"
#include "engine/models/breeze_tts/lora.h"
#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using engine::models::breeze_tts::load_breeze_lora_manifest;
using engine::models::breeze_tts::sha256_hex_bytes;
using engine::models::breeze_tts::sha256_hex_file;
using engine::models::breeze_tts::validate_breeze_lora_manifest;
using engine::test::require;

namespace {

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

template <class Fn>
void rejects(Fn fn, const char * message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, message);
}

}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_manifest_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    // Minimal fake adapter file for checksum wiring.
    std::vector<engine::io::SafeTensorWriteEntry> entries;
    engine::io::SafeTensorWriteEntry entry;
    entry.name = "lm_head.lora_A";
    entry.dtype = "F32";
    entry.shape = {8, 4};
    entry.data.assign(8 * 4 * sizeof(float), 0);
    entries.push_back(entry);
    entry.name = "lm_head.lora_B";
    entry.shape = {3, 8};
    entry.data.assign(3 * 8 * sizeof(float), 0);
    entries.push_back(entry);
    const auto adapter_path = root / "adapter.safetensors";
    engine::io::write_safetensors_file(adapter_path, entries);
    const auto adapter_hash = sha256_hex_file(adapter_path);

    const auto manifest_path = root / "adapter_config.json";
    write_text(
        manifest_path,
        std::string("{\n") +
            "  \"schema_version\": 1,\n"
            "  \"artifact_type\": \"breeze_lora_adapter\",\n"
            "  \"base_model\": {\n"
            "    \"id\": \"BreezeBlue/Breeze-TTS-2\",\n"
            "    \"revision\": \"abc123\",\n"
            "    \"files\": {\"config.json\": \"" + std::string(64, 'a') + "\"}\n"
            "  },\n"
            "  \"adapter\": {\n"
            "    \"file\": \"adapter.safetensors\",\n"
            "    \"sha256\": \"" + adapter_hash + "\"\n"
            "  },\n"
            "  \"lora\": {\n"
            "    \"variant\": \"backbone_depth_projection\",\n"
            "    \"rank\": 8,\n"
            "    \"alpha\": 16,\n"
            "    \"seed\": 7\n"
            "  }\n"
            "}\n");

    const auto manifest = load_breeze_lora_manifest(root);
    require(manifest.rank == 8, "rank");
    require(manifest.variant == "backbone_depth_projection", "variant");
    require(manifest.adapter_sha256 == adapter_hash, "adapter hash");

    auto bad = manifest;
    bad.rank = 4;
    rejects([&] { validate_breeze_lora_manifest(bad); }, "bad rank accepted");
    bad = manifest;
    bad.variant = "backbone_only";
    rejects([&] { validate_breeze_lora_manifest(bad); }, "bad variant accepted");

    write_text(manifest_path, "{ \"schema_version\": 2 }");
    rejects([&] { load_breeze_lora_manifest(root); }, "bad schema accepted");

    std::cout << "test_breeze_lora_manifest: ok\n";
    return 0;
}
