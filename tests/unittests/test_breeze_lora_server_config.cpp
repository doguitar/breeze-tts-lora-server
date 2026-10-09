#include "config.h"
#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using breeze_lora_server::is_loopback_host;
using breeze_lora_server::load_config;
using breeze_lora_server::save_config_atomically;
using breeze_lora_server::serialize_config;
using engine::test::require;

namespace {
void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

void write_minimal_wav(const std::filesystem::path & path) {
    const unsigned char wav[] = {
        'R','I','F','F', 36,0,0,0, 'W','A','V','E', 'f','m','t',' ',
        16,0,0,0, 1,0, 1,0, 0x40,0x1F,0,0, 0x80,0x3E,0,0, 2,0, 16,0,
        'd','a','t','a', 8,0,0,0, 0,0, 0,0, 0,0, 0,0
    };
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(wav), sizeof(wav));
}

template <class Fn>
void rejects(Fn fn, const char * msg) {
    bool threw = false;
    try { fn(); } catch (const std::exception &) { threw = true; }
    require(threw, msg);
}
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_server_config_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "base");
    std::filesystem::create_directories(root / "lora-a");
    const auto cfg = root / "server.json";
    write_text(cfg,
        "{\n"
        "  \"host\": \"127.0.0.1\",\n"
        "  \"port\": 8090,\n"
        "  \"backend\": \"cpu\",\n"
        "  \"max_queue_depth\": 2,\n"
        "  \"base_model\": \"base\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"default_instruction\": \"Calm.\"},\n"
        "    {\"id\": \"base-narrator\", \"lora\": null, \"default_instruction\": \"Narrate.\"}\n"
        "  ]\n"
        "}\n");
    auto loaded = load_config(cfg);
    require(loaded.port == 8090, "port");
    require(loaded.max_queue_depth == 2, "queue depth");
    require(loaded.voices.size() == 2, "voice count");
    require(loaded.voices[0].default_instruction == "Calm.", "default instruction");
    require(loaded.voices[0].lora.has_value(), "lora voice has path");
    require(!loaded.voices[1].lora.has_value(), "no-lora voice");
    require(!loaded.voices[0].voice_ref.has_value(), "voice_ref unset");
    require(loaded.config_dir == cfg.parent_path() ||
                std::filesystem::equivalent(loaded.config_dir, cfg.parent_path()),
            "config_dir");
    require(loaded.voices[0].lora_source == "lora-a", "lora source spelling");
    require(loaded.base_model_source == "base", "base_model source spelling");
    require(is_loopback_host(loaded.host), "loopback host 127.0.0.1");
    require(is_loopback_host("localhost"), "loopback host localhost");
    require(is_loopback_host("::1"), "loopback host ::1");
    require(!is_loopback_host("0.0.0.0"), "non-loopback 0.0.0.0");
    require(!loaded.management, "management defaults to false when absent");

    write_text(cfg,
        "{\n"
        "  \"host\": \"0.0.0.0\",\n"
        "  \"management\": true,\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": []\n"
        "}\n");
    const auto managed = load_config(cfg);
    require(managed.management, "management true is loaded");
    require(serialize_config(managed).find("\"management\": true") != std::string::npos,
            "serialize keeps management true");

    write_text(cfg,
        "{\n"
        "  \"management\": false,\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": []\n"
        "}\n");
    require(!load_config(cfg).management, "management false is loaded");

    write_text(cfg,
        "{\n"
        "  \"management\": \"yes\",\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": []\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "non-boolean management accepted");

    loaded.voices[0].default_instruction = "Updated calm.";
    save_config_atomically(loaded);
    const auto reloaded = load_config(cfg);
    require(reloaded.voices[0].default_instruction == "Updated calm.", "persisted instruction");
    require(reloaded.voices[0].lora_source == "lora-a", "relative lora preserved after save");
    require(reloaded.base_model_source == "base", "relative base_model preserved after save");
    const auto serialized = serialize_config(reloaded);
    require(serialized.find("\"voices\"") != std::string::npos, "serialize uses voices");
    require(serialized.find("\"lora\": \"lora-a\"") != std::string::npos, "serialize keeps relative lora");
    require(serialized.find("Updated calm.") != std::string::npos, "serialize includes new instruction");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": []\n"
        "}\n");
    const auto empty_voices = load_config(cfg);
    require(empty_voices.voices.empty(), "empty voices allowed");

    write_minimal_wav(root / "ref.wav");
    write_text(root / "ref.txt", "Some call me nature.\n");
    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": [\n"
        "    {\n"
        "      \"id\": \"adapter-a\",\n"
        "      \"lora\": \"lora-a\",\n"
        "      \"default_instruction\": \"Narrate calmly.\",\n"
        "      \"voice_ref\": \"ref.wav\",\n"
        "      \"reference_text_file\": \"ref.txt\"\n"
        "    }\n"
        "  ]\n"
        "}\n");
    const auto with_voice = load_config(cfg);
    require(with_voice.voices[0].voice_ref.has_value(), "voice_ref set");
    require(with_voice.voices[0].voice_ref->filename() == "ref.wav", "relative voice_ref resolved");
    require(std::filesystem::equivalent(*with_voice.voices[0].voice_ref, root / "ref.wav"), "voice_ref path");
    require(with_voice.voices[0].reference_text == "Some call me nature.", "reference_text_file loaded");
    require(with_voice.voices[0].default_instruction == "Narrate calmly.", "default instruction with voice");
    require(with_voice.voices[0].voice_ref_source == "ref.wav", "voice_ref source spelling");
    require(with_voice.voices[0].reference_text_file_source.has_value(), "reference_text_file source kept");
    require(*with_voice.voices[0].reference_text_file_source == "ref.txt", "reference_text_file spelling");
    const auto voice_serialized = serialize_config(with_voice);
    require(voice_serialized.find("\"reference_text_file\": \"ref.txt\"") != std::string::npos,
            "serialize emits reference_text_file when that was the source");
    require(voice_serialized.find("\"reference_text\":") == std::string::npos,
            "serialize omits inline reference_text when file source exists");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"voice_ref\": \"ref.wav\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "voice_ref without text accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"reference_text\": \"hi\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "reference_text without voice_ref accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"models\": [{\"id\": \"adapter-a\", \"lora\": \"lora-a\"}]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "legacy models schema accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\"},\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "duplicate ids accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"bad/id\", \"lora\": \"lora-a\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "voice id with slash accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"bad..id\", \"lora\": \"lora-a\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "voice id with .. accepted");

    std::cout << "test_breeze_lora_server_config: ok\n";
    return 0;
}
