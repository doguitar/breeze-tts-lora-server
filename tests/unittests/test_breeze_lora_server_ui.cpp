#include "config.h"
#include "runtime.h"
#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using breeze_lora_server::ServerConfig;
using breeze_lora_server::ServerRuntime;
using breeze_lora_server::load_config;
using engine::test::require;
using minitts::server::HttpRequest;
using minitts::server::HttpResponse;

namespace {

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

std::string minimal_wav_bytes() {
    const unsigned char wav[] = {
        'R','I','F','F', 36,0,0,0, 'W','A','V','E', 'f','m','t',' ',
        16,0,0,0, 1,0, 1,0, 0x40,0x1F,0,0, 0x80,0x3E,0,0, 2,0, 16,0,
        'd','a','t','a', 8,0,0,0, 0,0, 0,0, 0,0, 0,0
    };
    return std::string(reinterpret_cast<const char *>(wav), sizeof(wav));
}



HttpRequest make_request(
    const std::string & method,
    const std::string & path,
    const std::string & body = {},
    const std::string & content_type = {}) {
    HttpRequest request;
    request.method = method;
    request.path = path;
    request.body = body;
    if (!content_type.empty()) {
        request.headers["content-type"] = content_type;
    }
    return request;
}

std::string multipart_body(
    const std::string & boundary,
    const std::vector<std::pair<std::string, std::string>> & fields,
    const std::string & file_field,
    const std::string & filename,
    const std::string & file_bytes) {
    std::string body;
    for (const auto & [name, value] : fields) {
        body += "--" + boundary + "\r\n";
        body += "Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n";
        body += value + "\r\n";
    }
    if (!file_field.empty()) {
        body += "--" + boundary + "\r\n";
        body += "Content-Disposition: form-data; name=\"" + file_field +
            "\"; filename=\"" + filename + "\"\r\n";
        body += "Content-Type: audio/wav\r\n\r\n";
        body += file_bytes + "\r\n";
    }
    body += "--" + boundary + "--\r\n";
    return body;
}

bool body_contains(const HttpResponse & response, const std::string & needle) {
    return response.body.find(needle) != std::string::npos;
}

ServerConfig make_temp_config(const std::filesystem::path & root, const std::string & host) {
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "base");
    std::filesystem::create_directories(root / "lora-a");
    const auto cfg = root / "server.json";
    write_text(cfg,
        "{\n"
        "  \"host\": \"" + host + "\",\n"
        "  \"port\": 8091,\n"
        "  \"backend\": \"cpu\",\n"
        "  \"max_queue_depth\": 2,\n"
        "  \"base_model\": \"base\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"adapter-a\", \"lora\": \"lora-a\", \"default_instruction\": \"Calm.\"},\n"
        "    {\"id\": \"base-narrator\", \"lora\": null, \"default_instruction\": \"Narrate.\"}\n"
        "  ]\n"
        "}\n");
    return load_config(cfg);
}

void test_persistence_and_instruction_only() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_server_ui_persist";
    auto config = make_temp_config(root, "127.0.0.1");
    ServerRuntime runtime(std::move(config), ServerRuntime::ConfigOnlyInit{});

    auto update = make_request(
        "PUT",
        "/ui/voices/adapter-a",
        "{\"default_instruction\":\"Updated via UI.\",\"reference_text\":null,\"clear_reference\":false}",
        "application/json");
    auto response = runtime.handle(update);
    require(response.status == 200, "instruction update status");

    const auto reloaded = load_config(root / "server.json");
    require(reloaded.voices[0].default_instruction == "Updated via UI.", "persisted instruction via handler");
    require(reloaded.voices[0].lora_source == "lora-a", "relative lora preserved via handler");

    auto voices = runtime.handle(make_request("GET", "/ui/voices"));
    require(voices.status == 200, "ui voices status");
    require(body_contains(voices, "\"management_enabled\":true"), "management enabled on loopback");
    require(body_contains(voices, "Updated via UI."), "ui voices shows new instruction");

    // Temporary unsaved instruction on speech must not rewrite server.json.
    const auto before = std::filesystem::last_write_time(root / "server.json");
    auto speech = make_request(
        "POST",
        "/ui/audio/speech",
        "{\"model\":\"breeze-base\",\"voice\":\"adapter-a\",\"input\":\"hello\",\"instruction\":\"Temporary only.\",\"response_format\":\"wav\"}",
        "application/json");
    auto speech_response = runtime.handle(speech);
    // Config-only mode cannot synthesize; request is still accepted into the worker.
    require(speech_response.status == 500, "config-only speech returns synthesis error");
    require(std::filesystem::last_write_time(root / "server.json") == before, "temp instruction not persisted");
    const auto after_speech = load_config(root / "server.json");
    require(after_speech.voices[0].default_instruction == "Updated via UI.", "saved instruction unchanged");
}

void test_upload_and_empty_transcript() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_server_ui_upload";
    auto config = make_temp_config(root, "127.0.0.1");
    const auto cfg_path = root / "server.json";
    const auto cfg_before = std::filesystem::last_write_time(cfg_path);
    ServerRuntime runtime(std::move(config), ServerRuntime::ConfigOnlyInit{});

    const std::string boundary = "tesboundary";
    const auto empty_body = multipart_body(
        boundary,
        {{"reference_text", ""}, {"default_instruction", "Narrate."}},
        "reference_audio",
        "ref.wav",
        minimal_wav_bytes());
    auto empty = make_request(
        "POST",
        "/ui/voices/adapter-a/reference",
        empty_body,
        "multipart/form-data; boundary=" + boundary);
    auto empty_response = runtime.handle(empty);
    require(empty_response.status == 400, "empty transcript rejected");
    require(!std::filesystem::exists(root / "webui-references" / "adapter-a.wav"),
            "failed upload must not leave wav");
    require(std::filesystem::last_write_time(cfg_path) == cfg_before, "failed upload must not touch config");

    const auto ok_body = multipart_body(
        boundary,
        {{"reference_text", "Some call me nature."}, {"default_instruction", "Narrate calmly."}},
        "reference_audio",
        "ref.wav",
        minimal_wav_bytes());
    auto upload = make_request(
        "POST",
        "/ui/voices/adapter-a/reference",
        ok_body,
        "multipart/form-data; boundary=" + boundary);
    auto upload_response = runtime.handle(upload);
    require(upload_response.status == 200, "reference upload status");

    auto voices = runtime.handle(make_request("GET", "/ui/voices"));
    require(voices.status == 200, "ui voices after upload");
    require(body_contains(voices, "\"has_voice_ref\":true"), "has_voice_ref true after upload");
    require(body_contains(voices, "Narrate calmly."), "instruction updated by upload");
    require(body_contains(voices, "Some call me nature."), "transcript returned by ui voices");
    require(std::filesystem::exists(root / "webui-references" / "adapter-a.wav"), "upload wav persisted");
    require(!std::filesystem::exists(root / "webui-references" / "adapter-a.wav.tmp"),
            "upload temp wav removed after publish");

    const auto reloaded = load_config(cfg_path);
    require(reloaded.voices[0].voice_ref_source == "webui-references/adapter-a.wav",
            "relative webui voice_ref spelling");
    require(!reloaded.voices[0].reference_text_file_source.has_value(), "inline reference_text after upload");
    require(reloaded.voices[0].reference_text == "Some call me nature.", "inline transcript persisted");
}

void test_url_encoded_voice_id() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_server_ui_encoded_id";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "base");
    std::filesystem::create_directories(root / "lora spaced");
    const auto cfg = root / "server.json";
    write_text(cfg,
        "{\n"
        "  \"host\": \"127.0.0.1\",\n"
        "  \"port\": 8092,\n"
        "  \"backend\": \"cpu\",\n"
        "  \"max_queue_depth\": 2,\n"
        "  \"base_model\": \"base\",\n"
        "  \"voices\": [\n"
        "    {\"id\": \"adapter a\", \"lora\": \"lora spaced\", \"default_instruction\": \"Calm.\"}\n"
        "  ]\n"
        "}\n");
    ServerRuntime runtime(load_config(cfg), ServerRuntime::ConfigOnlyInit{});

    auto update = make_request(
        "PUT",
        "/ui/voices/adapter%20a",
        "{\"default_instruction\":\"Encoded path works.\",\"reference_text\":null,\"clear_reference\":false}",
        "application/json");
    auto response = runtime.handle(update);
    require(response.status == 200, "encoded voice id update status");

    const auto reloaded = load_config(cfg);
    require(reloaded.voices[0].id == "adapter a", "spaced voice id intact");
    require(reloaded.voices[0].default_instruction == "Encoded path works.",
            "persisted instruction for encoded voice id");

    const std::string boundary = "encboundary";
    const auto body = multipart_body(
        boundary,
        {{"reference_text", "Spaced voice transcript."}, {"default_instruction", "Narrate spaced."}},
        "reference_audio",
        "ref.wav",
        minimal_wav_bytes());
    auto upload = make_request(
        "POST",
        "/ui/voices/adapter%20a/reference",
        body,
        "multipart/form-data; boundary=" + boundary);
    auto upload_response = runtime.handle(upload);
    require(upload_response.status == 200, "encoded voice id reference upload");
    require(std::filesystem::exists(root / "webui-references" / "adapter a.wav"),
            "upload wav uses decoded voice id");
    require(!std::filesystem::exists(root / "webui-references" / "adapter a.wav.tmp"),
            "upload temp wav removed after publish");
}

void test_access_non_loopback() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_server_ui_access";
    auto config = make_temp_config(root, "0.0.0.0");
    const auto cfg_path = root / "server.json";
    const auto before = std::filesystem::last_write_time(cfg_path);
    ServerRuntime runtime(std::move(config), ServerRuntime::ConfigOnlyInit{});

    auto index = runtime.handle(make_request("GET", "/"));
    require(index.status == 200, "GET / on non-loopback");
    require(index.content_type.find("text/html") != std::string::npos, "index content type");
    require(!index.body.empty(), "index body non-empty");
    require(body_contains(index, "id=\"voice\""), "ui labels voice selector");
    require(body_contains(index, "/ui/voices"), "ui fetches voices endpoint");

    auto voices = runtime.handle(make_request("GET", "/ui/voices"));
    require(voices.status == 200, "GET /ui/voices on non-loopback");
    require(body_contains(voices, "\"management_enabled\":false"), "management disabled");

    auto update = make_request(
        "PUT",
        "/ui/voices/adapter-a",
        "{\"default_instruction\":\"Should fail.\",\"reference_text\":null,\"clear_reference\":false}",
        "application/json");
    auto update_response = runtime.handle(update);
    require(update_response.status == 403, "PUT forbidden off loopback");
    require(body_contains(update_response, "loopback"), "403 mentions loopback");

    const std::string boundary = "denyboundary";
    const auto body = multipart_body(
        boundary,
        {{"reference_text", "Nope."}, {"default_instruction", "Nope."}},
        "reference_audio",
        "ref.wav",
        minimal_wav_bytes());
    auto upload = make_request(
        "POST",
        "/ui/voices/adapter-a/reference",
        body,
        "multipart/form-data; boundary=" + boundary);
    auto upload_response = runtime.handle(upload);
    require(upload_response.status == 403, "POST reference forbidden off loopback");
    require(!std::filesystem::exists(root / "webui-references"), "no webui-references on 403");
    require(std::filesystem::last_write_time(cfg_path) == before, "server.json unchanged on 403");
}

void test_api_model_and_voice_selection() {
    const auto root = std::filesystem::temp_directory_path() / "breeze_lora_server_ui_api";
    auto config = make_temp_config(root, "127.0.0.1");
    ServerRuntime runtime(std::move(config), ServerRuntime::ConfigOnlyInit{});

    auto models = runtime.handle(make_request("GET", "/v1/models"));
    require(models.status == 200, "/v1/models status");
    require(body_contains(models, "\"id\":\"breeze-base\""), "singleton breeze-base id");
    require(body_contains(models, "\"owned_by\":\"breeze-lora-server\""), "owned_by present");
    require(!body_contains(models, "adapter-a"), "adapter id not listed as model");
    require(!body_contains(models, "default_instruction"), "voice metadata absent from /v1/models");

    auto base_speech = runtime.handle(make_request(
        "POST",
        "/v1/audio/speech",
        "{\"model\":\"breeze-base\",\"input\":\"hello\",\"response_format\":\"wav\"}",
        "application/json"));
    require(base_speech.status == 500, "no-voice request reaches config-only synthesis");
    require(body_contains(base_speech, "synthesis unavailable in config-only mode"),
            "no-voice fails with unavailable synthesis");

    auto voice_speech = runtime.handle(make_request(
        "POST",
        "/v1/audio/speech",
        "{\"model\":\"breeze-base\",\"voice\":\"adapter-a\",\"input\":\"hello\",\"response_format\":\"wav\"}",
        "application/json"));
    require(voice_speech.status == 500, "configured voice reaches config-only synthesis");
    require(body_contains(voice_speech, "synthesis unavailable in config-only mode"),
            "configured voice fails with unavailable synthesis");

    auto no_lora_voice = runtime.handle(make_request(
        "POST",
        "/v1/audio/speech",
        "{\"model\":\"breeze-base\",\"voice\":\"base-narrator\",\"input\":\"hello\",\"response_format\":\"wav\"}",
        "application/json"));
    require(no_lora_voice.status == 500, "no-lora voice reaches config-only synthesis");

    auto adapter_as_model = runtime.handle(make_request(
        "POST",
        "/v1/audio/speech",
        "{\"model\":\"adapter-a\",\"input\":\"hello\",\"response_format\":\"wav\"}",
        "application/json"));
    require(adapter_as_model.status == 400, "adapter id in model rejected");
    require(body_contains(adapter_as_model, "unknown model id: adapter-a"),
            "adapter-as-model error text");

    auto unknown_voice = runtime.handle(make_request(
        "POST",
        "/v1/audio/speech",
        "{\"model\":\"breeze-base\",\"voice\":\"missing\",\"input\":\"hello\",\"response_format\":\"wav\"}",
        "application/json"));
    require(unknown_voice.status == 400, "unknown voice rejected");
    require(body_contains(unknown_voice, "unknown voice id: missing"), "unknown voice error text");
}

}  // namespace

int main() {
    test_persistence_and_instruction_only();
    test_upload_and_empty_transcript();
    test_url_encoded_voice_id();
    test_access_non_loopback();
    test_api_model_and_voice_selection();
    std::cout << "test_breeze_lora_server_ui: ok\n";
    return 0;
}
