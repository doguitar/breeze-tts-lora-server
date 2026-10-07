#pragma once

#include "config.h"
#include "http.h"

#include "engine/framework/core/execution_context.h"
#include "engine/models/breeze_tts/assets.h"
#include "engine/models/breeze_tts/generator.h"
#include "engine/models/breeze_tts/speech_decoder.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace breeze_lora_server {

enum class ResponseFormat { Wav, Mp3 };

class ServerRuntime final : public minitts::server::IHttpHandler {
public:
    struct ConfigOnlyInit {};

    explicit ServerRuntime(ServerConfig config);
    // Skips GGUF/generator load so UI config routes can be unit-tested without model weights.
    ServerRuntime(ServerConfig config, ConfigOnlyInit);
    ~ServerRuntime() override;

    minitts::server::HttpResponse handle(const minitts::server::HttpRequest & request) override;
    void request_shutdown();

private:
    struct ModelDefaults {
        std::string default_instruction = "Speak clearly and naturally.";
        std::string reference_text;
        std::optional<engine::models::breeze_tts::BreezeSpeechCodes> reference_codes;
    };

    struct Job {
        std::string model;
        std::string text;
        std::string instruction;
        std::string reference_text;
        std::optional<engine::models::breeze_tts::BreezeSpeechCodes> reference_codes;
        std::optional<engine::runtime::AudioBuffer> reference_audio;
        ResponseFormat response_format = ResponseFormat::Wav;
        uint64_t seed = 0;
        float guidance_scale = 1.0F;
        float temperature = 0.9F;
        float depth_temperature = 0.9F;
        int64_t top_k = 50;
        float top_p = 1.0F;
        int64_t max_tokens = 1500;
        std::promise<minitts::server::HttpResponse> response;
    };

    void init_model_maps();
    void start_worker();

    minitts::server::HttpResponse handle_health() const;
    minitts::server::HttpResponse handle_models() const;
    minitts::server::HttpResponse handle_speech(const std::string & body_text);
    minitts::server::HttpResponse handle_ui_index() const;
    minitts::server::HttpResponse handle_ui_models() const;
    minitts::server::HttpResponse handle_ui_model_update(
        const minitts::server::HttpRequest & request,
        const std::string & model_id);
    minitts::server::HttpResponse handle_ui_reference_upload(
        const minitts::server::HttpRequest & request,
        const std::string & model_id);
    minitts::server::HttpResponse handle_ui_speech(const minitts::server::HttpRequest & request);
    bool ui_management_enabled() const;
    minitts::server::HttpResponse enqueue_job(std::shared_ptr<Job> job);
    void worker_loop();
    void activate_model(const std::string & model_id);
    minitts::server::HttpResponse synthesize(const Job & job);
    ServerConfig config_;
    std::unordered_map<std::string, std::optional<std::filesystem::path>> model_paths_;
    std::unordered_map<std::string, ModelDefaults> model_defaults_;
    std::unordered_set<std::string> activated_once_;
    std::unique_ptr<engine::core::ExecutionContext> execution_;
    std::shared_ptr<const engine::models::breeze_tts::BreezeTTSAssets> assets_;
    std::unique_ptr<engine::models::breeze_tts::BreezeGeneratorRuntime> generator_;

    mutable std::mutex defaults_mutex_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
};

}  // namespace breeze_lora_server
