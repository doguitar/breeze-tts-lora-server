#include "config.h"
#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using breeze_lora_server::load_config;
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
        "  \"models\": [\n"
        "    {\"id\": \"breeze-base\", \"lora\": null},\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"default_instruction\": \"Calm.\"}\n"
        "  ]\n"
        "}\n");
    const auto loaded = load_config(cfg);
    require(loaded.port == 8090, "port");
    require(loaded.max_queue_depth == 2, "queue depth");
    require(loaded.models.size() == 2, "model count");
    require(loaded.models[1].default_instruction == "Calm.", "default instruction");
    require(!loaded.models[1].voice_ref.has_value(), "voice_ref unset");
    require(loaded.config_dir == cfg.parent_path(), "config_dir");

    write_minimal_wav(root / "ref.wav");
    write_text(root / "ref.txt", "Some call me nature.\n");
    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"models\": [\n"
        "    {\"id\": \"breeze-base\", \"lora\": null},\n"
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
    require(with_voice.models[1].voice_ref.has_value(), "voice_ref set");
    require(with_voice.models[1].voice_ref->filename() == "ref.wav", "relative voice_ref resolved");
    require(std::filesystem::equivalent(*with_voice.models[1].voice_ref, root / "ref.wav"), "voice_ref path");
    require(with_voice.models[1].reference_text == "Some call me nature.", "reference_text_file loaded");
    require(with_voice.models[1].default_instruction == "Narrate calmly.", "default instruction with voice");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"models\": [\n"
        "    {\"id\": \"breeze-base\", \"lora\": null},\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"voice_ref\": \"ref.wav\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "voice_ref without text accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"backend\": \"cpu\",\n"
        "  \"models\": [\n"
        "    {\"id\": \"breeze-base\", \"lora\": null},\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"reference_text\": \"hi\"}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "reference_text without voice_ref accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"models\": [{\"id\": \"adapter-a\", \"lora\": \"lora-a\"}]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "missing breeze-base accepted");

    write_text(cfg,
        "{\n"
        "  \"base_model\": \"base\",\n"
        "  \"models\": [\n"
        "    {\"id\": \"breeze-base\", \"lora\": null},\n"
        "    {\"id\": \"breeze-base\", \"lora\": null}\n"
        "  ]\n"
        "}\n");
    rejects([&] { load_config(cfg); }, "duplicate ids accepted");

    std::cout << "test_breeze_lora_server_config: ok\n";
    return 0;
}
