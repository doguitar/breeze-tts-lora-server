#include "runtime.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/models/breeze_tts/lora.h"

#include <lame.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace breeze_lora_server {
namespace {

using engine::models::breeze_tts::BreezeGenerationRequest;
using engine::models::breeze_tts::BreezeGeneratorRuntime;
using engine::models::breeze_tts::BreezeLoraAdapterTensors;
using engine::models::breeze_tts::kBreezeBaseModelId;
using engine::models::breeze_tts::load_breeze_lora_adapter;
using engine::models::breeze_tts::load_breeze_tts_assets;
using minitts::server::HttpRequest;
using minitts::server::HttpResponse;
using minitts::server::error_response;
using minitts::server::json_response;

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "cpu") return engine::core::BackendType::Cpu;
    throw std::runtime_error("unsupported backend: " + value);
}

engine::runtime::AudioBuffer load_voice_ref(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
        throw std::runtime_error("invalid voice_ref audio: " + path.string());
    }
    return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

std::vector<uint8_t> encode_pcm16_wav(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("invalid audio buffer metadata");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio sample count must be divisible by channels");
    }
    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint16_t bits_per_sample = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t byte_rate = static_cast<uint32_t>(audio.sample_rate) * channels * bits_per_sample / 8;
    const uint16_t block_align = static_cast<uint16_t>(channels * bits_per_sample / 8);
    std::vector<uint8_t> out;
    out.reserve(44 + data_bytes);
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    auto append_u16 = [&](uint16_t value) { append_bytes(&value, sizeof(value)); };
    auto append_u32 = [&](uint32_t value) { append_bytes(&value, sizeof(value)); };
    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_u32(riff_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(16);
    append_u16(1);
    append_u16(channels);
    append_u32(static_cast<uint32_t>(audio.sample_rate));
    append_u32(byte_rate);
    append_u16(block_align);
    append_u16(bits_per_sample);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_u32(data_bytes);
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

std::vector<uint8_t> encode_mp3(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("invalid audio buffer metadata");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio sample count must be divisible by channels");
    }

    std::vector<int16_t> pcm;
    pcm.reserve(audio.samples.size());
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        pcm.push_back(static_cast<int16_t>(std::lrint(sample * 32767.0F)));
    }

    lame_t lame = lame_init();
    if (lame == nullptr) {
        throw std::runtime_error("lame_init failed");
    }

    const int channels = audio.channels;
    lame_set_in_samplerate(lame, audio.sample_rate);
    lame_set_num_channels(lame, channels);
    lame_set_mode(lame, channels <= 1 ? MONO : STEREO);
    lame_set_VBR(lame, vbr_off);
    lame_set_brate(lame, 128);
    lame_set_quality(lame, 5);
    if (lame_init_params(lame) < 0) {
        lame_close(lame);
        throw std::runtime_error("lame_init_params failed");
    }

    const size_t frames = pcm.size() / static_cast<size_t>(channels);
    // LAME recommends ~1.25 * samples + 7200 for the MP3 output buffer.
    const size_t mp3_buf_size = static_cast<size_t>(1.25 * static_cast<double>(pcm.size())) + 7200;
    std::vector<uint8_t> mp3_buf(mp3_buf_size);
    std::vector<uint8_t> out;

    constexpr size_t kChunkFrames = 1152;
    size_t frame_offset = 0;
    while (frame_offset < frames) {
        const size_t chunk_frames = std::min(kChunkFrames, frames - frame_offset);
        const int16_t * chunk = pcm.data() + frame_offset * static_cast<size_t>(channels);
        const int written = (channels == 1)
            ? lame_encode_buffer(
                  lame,
                  chunk,
                  nullptr,
                  static_cast<int>(chunk_frames),
                  mp3_buf.data(),
                  static_cast<int>(mp3_buf.size()))
            : lame_encode_buffer_interleaved(
                  lame,
                  const_cast<int16_t *>(chunk),
                  static_cast<int>(chunk_frames),
                  mp3_buf.data(),
                  static_cast<int>(mp3_buf.size()));
        if (written < 0) {
            lame_close(lame);
            throw std::runtime_error("lame_encode failed");
        }
        out.insert(out.end(), mp3_buf.begin(), mp3_buf.begin() + written);
        frame_offset += chunk_frames;
    }

    const int flushed = lame_encode_flush(
        lame, mp3_buf.data(), static_cast<int>(mp3_buf.size()));
    if (flushed < 0) {
        lame_close(lame);
        throw std::runtime_error("lame_encode_flush failed");
    }
    out.insert(out.end(), mp3_buf.begin(), mp3_buf.begin() + flushed);
    lame_close(lame);
    return out;
}

std::string json_quote(const std::string & value) {
    return engine::io::json::stringify_string(value);
}

}  // namespace

ServerRuntime::ServerRuntime(ServerConfig config) : config_(std::move(config)) {
    for (const auto & entry : config_.models) {
        model_paths_.emplace(entry.id, entry.lora);
    }

    engine::core::BackendConfig backend_config;
    backend_config.type = parse_backend(config_.backend);
    backend_config.device = config_.device;
    backend_config.threads = config_.threads;
    execution_ = std::make_unique<engine::core::ExecutionContext>(backend_config);
    assets_ = load_breeze_tts_assets(config_.base_model);
    std::vector<std::pair<std::string, std::filesystem::path>> lora_adapters;
    for (const auto & entry : config_.models) {
        if (entry.lora.has_value()) {
            lora_adapters.emplace_back(entry.id, *entry.lora);
        }
    }
    generator_ = std::make_unique<BreezeGeneratorRuntime>(
        assets_,
        *execution_,
        1024ull * 1024ull * 1024ull,
        2048ull * 1024ull * 1024ull,
        engine::assets::TensorStorageType::Native,
        engine::core::AttentionPreference::Auto,
        engine::models::breeze_tts::Bf16ActivationMode::Auto,
        std::move(lora_adapters),
        config_.base_revision);

    auto & lora = generator_->lora_manager();
    lora.activate_base();
    activated_once_.insert(kBreezeBaseModelId);

    for (const auto & entry : config_.models) {
        ModelDefaults defaults;
        defaults.default_instruction = entry.default_instruction;
        defaults.reference_text = entry.reference_text;
        if (entry.voice_ref.has_value()) {
            const auto audio = load_voice_ref(*entry.voice_ref);
            defaults.reference_codes = generator_->encode_reference(audio);
        }
        model_defaults_.emplace(entry.id, std::move(defaults));
    }

    worker_ = std::thread([this] { worker_loop(); });
}

ServerRuntime::~ServerRuntime() {
    request_shutdown();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void ServerRuntime::request_shutdown() {
    stopping_.store(true);
    cv_.notify_all();
    std::deque<std::shared_ptr<Job>> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending.swap(queue_);
    }
    for (auto & job : pending) {
        job->response.set_value(error_response(503, "server shutting down", "server_error"));
    }
}

HttpResponse ServerRuntime::handle(const HttpRequest & request) {
    try {
        if (request.method == "GET" && request.path == "/health") {
            return handle_health();
        }
        if (request.method == "GET" && request.path == "/v1/models") {
            return handle_models();
        }
        if (request.method == "POST" && request.path == "/v1/audio/speech") {
            return handle_speech(request.body);
        }
        return error_response(404, "not found", "invalid_request_error");
    } catch (const std::exception & ex) {
        return error_response(400, ex.what(), "invalid_request_error");
    }
}

HttpResponse ServerRuntime::handle_health() const {
    return json_response("{\"status\":\"ok\"}");
}

HttpResponse ServerRuntime::handle_models() const {
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    bool first = true;
    for (const auto & entry : config_.models) {
        if (!first) out << ',';
        first = false;
        out << "{\"id\":" << json_quote(entry.id)
            << ",\"object\":\"model\",\"owned_by\":\"breeze-lora-server\""
            << ",\"default_instruction\":" << json_quote(entry.default_instruction)
            << ",\"has_voice_ref\":" << (entry.voice_ref.has_value() ? "true" : "false")
            << "}";
    }
    out << "]}";
    return json_response(out.str());
}

HttpResponse ServerRuntime::handle_speech(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    if (body.find("stream") != nullptr || body.find("stream_format") != nullptr) {
        return error_response(400, "streaming is not supported", "invalid_request_error");
    }
    ResponseFormat parsed_format = ResponseFormat::Wav;
    if (body.find("response_format") != nullptr) {
        const auto response_format = engine::io::json::require_string(body, "response_format");
        if (response_format == "wav") {
            parsed_format = ResponseFormat::Wav;
        } else if (response_format == "mp3") {
            parsed_format = ResponseFormat::Mp3;
        } else {
            return error_response(
                400,
                "only response_format=wav or mp3 is supported",
                "invalid_request_error");
        }
    }
    const auto model = engine::io::json::require_string(body, "model");
    if (!model_paths_.count(model)) {
        return error_response(400, "unknown model id: " + model, "invalid_request_error");
    }
    const auto input = engine::io::json::require_string(body, "input");
    if (input.empty()) {
        return error_response(400, "input must be non-empty", "invalid_request_error");
    }

    const auto defaults_it = model_defaults_.find(model);
    if (defaults_it == model_defaults_.end()) {
        return error_response(500, "missing model defaults for: " + model, "server_error");
    }
    const auto & defaults = defaults_it->second;

    auto job = std::make_shared<Job>();
    job->model = model;
    job->text = input;
    job->response_format = parsed_format;

    const bool has_top_level_instruction = body.find("instruction") != nullptr;
    job->instruction = engine::io::json::optional_string(body, "instruction", defaults.default_instruction);
    if (const auto * options = body.find("options"); options != nullptr && options->is_object()) {
        if (options->find("instruction") != nullptr) {
            job->instruction = engine::io::json::optional_string(*options, "instruction", job->instruction);
        } else if (!has_top_level_instruction) {
            job->instruction = defaults.default_instruction;
        }
    }
    if (job->instruction.empty()) {
        return error_response(400, "instruction must be non-empty", "invalid_request_error");
    }

    const auto * req_voice_ref = body.find("voice_ref");
    const auto * req_reference_text = body.find("reference_text");
    const bool has_req_voice = req_voice_ref != nullptr && !req_voice_ref->is_null();
    const bool has_req_text = req_reference_text != nullptr && !req_reference_text->is_null();
    if (has_req_voice != has_req_text) {
        return error_response(
            400,
            "voice_ref and reference_text must be provided together",
            "invalid_request_error");
    }
    if (has_req_voice) {
        std::filesystem::path voice_path(req_voice_ref->as_string());
        if (!voice_path.is_absolute()) {
            voice_path = (config_.config_dir / voice_path).lexically_normal();
        }
        const auto text = req_reference_text->as_string();
        if (text.empty()) {
            return error_response(400, "reference_text must be non-empty", "invalid_request_error");
        }
        job->reference_audio = load_voice_ref(voice_path);
        job->reference_text = text;
    } else if (defaults.reference_codes.has_value()) {
        job->reference_codes = defaults.reference_codes;
        job->reference_text = defaults.reference_text;
    }

    job->seed = static_cast<uint64_t>(engine::io::json::optional_i64(body, "seed", 0));
    job->guidance_scale = engine::io::json::optional_f32(body, "guidance_scale", 1.0F);
    job->temperature = engine::io::json::optional_f32(body, "temperature", 0.9F);
    job->depth_temperature = engine::io::json::optional_f32(body, "depth_temperature", 0.9F);
    job->top_k = engine::io::json::optional_i64(body, "top_k", 50);
    job->top_p = engine::io::json::optional_f32(body, "top_p", 1.0F);
    job->max_tokens = engine::io::json::optional_i64(body, "max_tokens", 1500);

    auto future = job->response.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(queue_.size()) >= config_.max_queue_depth) {
            return error_response(503, "queue_full", "server_error");
        }
        queue_.push_back(job);
    }
    cv_.notify_one();
    return future.get();
}

void ServerRuntime::activate_model(const std::string & model_id) {
    if (model_id == kBreezeBaseModelId) {
        generator_->lora_manager().activate_base();
        return;
    }
    generator_->lora_manager().activate_adapter(model_id);
}

HttpResponse ServerRuntime::synthesize(const Job & job) {
    using clock = std::chrono::steady_clock;
    const auto ms_since = [](clock::time_point start) {
        return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    };
    const auto format_ms = [](double value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(1) << value;
        return out.str();
    };

    const bool first_load = !activated_once_.count(job.model);
    double load_ms = 0.0;
    double generate_ms = 0.0;

    try {
        if (generator_->lora_manager().active_id() != job.model) {
            const auto load_started = clock::now();
            activate_model(job.model);
            load_ms = ms_since(load_started);
            activated_once_.insert(job.model);
        }

        BreezeGenerationRequest request;
        request.text = job.text;
        request.instruction = job.instruction;
        request.reference_text = job.reference_text;
        request.reference_codes = job.reference_codes;
        request.reference_audio = job.reference_audio;
        request.seed = job.seed;
        request.guidance_scale = job.guidance_scale;
        request.temperature = job.temperature;
        request.depth_temperature = job.depth_temperature;
        request.top_k = job.top_k;
        request.top_p = job.top_p;
        request.max_tokens = job.max_tokens;

        const auto generate_started = clock::now();
        const auto audio = generator_->generate(request);
        generate_ms = ms_since(generate_started);
        HttpResponse response;
        response.status = 200;
        if (job.response_format == ResponseFormat::Mp3) {
            const auto mp3 = encode_mp3(audio);
            response.content_type = "audio/mpeg";
            response.body.assign(reinterpret_cast<const char *>(mp3.data()), mp3.size());
        } else {
            const auto wav = encode_pcm16_wav(audio);
            response.content_type = "audio/wav";
            response.body.assign(reinterpret_cast<const char *>(wav.data()), wav.size());
        }
        engine::debug::log_message(
            engine::debug::LogLevel::Info,
            "breeze_lora_server",
            "speech request model=" + job.model
                + " first_load=" + (first_load ? "true" : "false")
                + " load_ms=" + format_ms(load_ms)
                + " generate_ms=" + format_ms(generate_ms)
                + " status=ok");
        return response;
    } catch (const std::exception &) {
        engine::debug::log_message(
            engine::debug::LogLevel::Error,
            "breeze_lora_server",
            "speech request model=" + job.model
                + " first_load=" + (first_load ? "true" : "false")
                + " load_ms=" + format_ms(load_ms)
                + " generate_ms=" + format_ms(generate_ms)
                + " status=error");
        throw;
    }
}

void ServerRuntime::worker_loop() {
    while (true) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_.load() || !queue_.empty(); });
            if (stopping_.load() && queue_.empty()) {
                return;
            }
            job = queue_.front();
            queue_.pop_front();
        }
        try {
            job->response.set_value(synthesize(*job));
        } catch (const std::exception & ex) {
            job->response.set_value(error_response(500, ex.what(), "server_error"));
        }
    }
}

}  // namespace breeze_lora_server
