#include "runtime.h"

#include "multipart.h"
#include "ui_assets.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/models/breeze_tts/lora.h"

#include <lame.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace breeze_lora_server {
namespace {

using engine::models::breeze_tts::BreezeGenerationRequest;
using engine::models::breeze_tts::BreezeGeneratorRuntime;
using engine::models::breeze_tts::kBreezeBaseModelId;
using engine::models::breeze_tts::load_breeze_tts_assets;
using minitts::server::HttpRequest;
using minitts::server::HttpResponse;
using minitts::server::MultipartPart;
using minitts::server::error_response;
using minitts::server::extract_multipart_boundary;
using minitts::server::json_response;
using minitts::server::parse_multipart_body;

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

engine::runtime::AudioBuffer load_voice_ref_bytes(std::string_view bytes, const std::string & label) {
    const auto wav = engine::audio::read_wav_f32(bytes);
    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
        throw std::runtime_error("invalid voice_ref audio: " + label);
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

void log_non_success_request(const HttpRequest & request, const HttpResponse & response) {
    if (response.status == 200) {
        return;
    }

    std::ostringstream headers;
    headers << '{';
    bool first = true;
    for (const auto & [key, value] : request.headers) {
        if (!first) {
            headers << ',';
        }
        first = false;
        headers << json_quote(key) << ':' << json_quote(value);
    }
    headers << '}';

    std::ostringstream message;
    message << "non-success response"
            << " status=" << response.status
            << " method=" << request.method
            << " path=" << request.path
            << " query=" << json_quote(request.query)
            << " headers=" << headers.str()
            << " body=" << json_quote(request.body)
            << " response_body=" << json_quote(response.body);
    engine::debug::log_message(
        engine::debug::LogLevel::Error,
        "breeze_lora_server",
        message.str());
}

std::string request_content_type(const HttpRequest & request) {
    const auto it = request.headers.find("content-type");
    return it == request.headers.end() ? std::string() : it->second;
}

const MultipartPart * find_part(const std::vector<MultipartPart> & parts, const std::string & name) {
    for (const auto & part : parts) {
        if (part.name == name) {
            return &part;
        }
    }
    return nullptr;
}

int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return -1;
}

// Percent-decode a path segment. Unlike form decoding, '+' stays literal because
// encodeURIComponent emits '%20' for spaces.
std::string percent_decode_path_segment(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '%' && i + 2 < value.size()) {
            const int hi = hex_value(value[i + 1]);
            const int lo = hex_value(value[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(value[i]);
    }
    return out;
}

std::optional<std::string> ui_voice_id_from_path(const std::string & path, bool reference_suffix) {
    static constexpr std::string_view kPrefix = "/ui/voices/";
    if (path.rfind(kPrefix.data(), 0) != 0) {
        return std::nullopt;
    }
    const std::string rest = path.substr(kPrefix.size());
    if (rest.empty()) {
        return std::nullopt;
    }
    std::string encoded_id;
    if (reference_suffix) {
        static constexpr std::string_view kSuffix = "/reference";
        if (rest.size() <= kSuffix.size() || rest.compare(rest.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
            return std::nullopt;
        }
        encoded_id = rest.substr(0, rest.size() - kSuffix.size());
    } else {
        if (rest.find('/') != std::string::npos) {
            return std::nullopt;
        }
        encoded_id = rest;
    }
    if (encoded_id.empty()) {
        return std::nullopt;
    }
    return percent_decode_path_segment(encoded_id);
}

void write_binary_file(const std::filesystem::path & path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
        throw std::runtime_error("failed to flush file: " + path.string());
    }
}

}  // namespace

void ServerRuntime::init_voice_maps() {
    for (const auto & entry : config_.voices) {
        if (entry.lora.has_value()) {
            voice_lora_paths_.emplace(entry.id, *entry.lora);
        }
        VoiceDefaults defaults;
        defaults.default_instruction = entry.default_instruction;
        defaults.reference_text = entry.reference_text;
        voice_defaults_.emplace(entry.id, std::move(defaults));
    }
}

std::string ServerRuntime::activation_id_for_voice(const std::string & voice_id) const {
    if (voice_id.empty() || !voice_lora_paths_.count(voice_id)) {
        return kBreezeBaseModelId;
    }
    return voice_id;
}

std::optional<HttpResponse> ServerRuntime::resolve_voice_selection(
    const std::string & model,
    const std::optional<std::string> & voice,
    Job & job) const {
    if (model != kPublicModelId) {
        return error_response(400, "unknown model id: " + model, "invalid_request_error");
    }
    if (!voice.has_value()) {
        job.voice.clear();
        return std::nullopt;
    }
    if (voice->empty()) {
        return error_response(400, "unknown voice id: " + *voice, "invalid_request_error");
    }
    if (!voice_defaults_.count(*voice)) {
        return error_response(400, "unknown voice id: " + *voice, "invalid_request_error");
    }
    job.voice = *voice;
    return std::nullopt;
}

void ServerRuntime::start_worker() {
    worker_ = std::thread([this] { worker_loop(); });
}

ServerRuntime::ServerRuntime(ServerConfig config) : config_(std::move(config)) {
    init_voice_maps();

    engine::core::BackendConfig backend_config;
    backend_config.type = parse_backend(config_.backend);
    backend_config.device = config_.device;
    backend_config.threads = config_.threads;
    execution_ = std::make_unique<engine::core::ExecutionContext>(backend_config);
    assets_ = load_breeze_tts_assets(config_.base_model);
    std::vector<std::pair<std::string, std::filesystem::path>> lora_adapters;
    for (const auto & entry : config_.voices) {
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

    for (const auto & entry : config_.voices) {
        auto & defaults = voice_defaults_.at(entry.id);
        if (entry.voice_ref.has_value()) {
            const auto audio = load_voice_ref(*entry.voice_ref);
            defaults.reference_codes = generator_->encode_reference(audio);
        }
    }

    start_worker();
}

ServerRuntime::ServerRuntime(ServerConfig config, ConfigOnlyInit) : config_(std::move(config)) {
    init_voice_maps();
    start_worker();
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

bool ServerRuntime::ui_management_enabled() const {
    return is_loopback_host(config_.host);
}

HttpResponse ServerRuntime::enqueue_job(std::shared_ptr<Job> job) {
    auto future = job->response.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(queue_.size()) >= config_.max_queue_depth) {
            return error_response(503, "queue_full", "server_error");
        }
        queue_.push_back(std::move(job));
    }
    cv_.notify_one();
    return future.get();
}

HttpResponse ServerRuntime::handle(const HttpRequest & request) {
    HttpResponse response;
    try {
        if (request.method == "GET" && request.path == "/health") {
            response = handle_health();
        } else if (request.method == "GET" && request.path == "/v1/models") {
            response = handle_models();
        } else if (request.method == "POST" && request.path == "/v1/audio/speech") {
            response = handle_speech(request.body);
        } else if (request.method == "GET" && request.path == "/") {
            response = handle_ui_index();
        } else if (request.method == "GET" && request.path == "/ui/voices") {
            response = handle_ui_voices();
        } else if (request.method == "PUT") {
            if (const auto voice_id = ui_voice_id_from_path(request.path, false)) {
                response = handle_ui_voice_update(request, *voice_id);
            } else {
                response = error_response(404, "not found", "invalid_request_error");
            }
        } else if (request.method == "POST" && request.path == "/ui/audio/speech") {
            response = handle_ui_speech(request);
        } else if (request.method == "POST") {
            if (const auto voice_id = ui_voice_id_from_path(request.path, true)) {
                response = handle_ui_reference_upload(request, *voice_id);
            } else {
                response = error_response(404, "not found", "invalid_request_error");
            }
        } else {
            response = error_response(404, "not found", "invalid_request_error");
        }
    } catch (const std::exception & ex) {
        response = error_response(400, ex.what(), "invalid_request_error");
    }
    log_non_success_request(request, response);
    return response;
}

HttpResponse ServerRuntime::handle_health() const {
    return json_response("{\"status\":\"ok\"}");
}

HttpResponse ServerRuntime::handle_models() const {
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":["
        << "{\"id\":" << json_quote(kPublicModelId)
        << ",\"object\":\"model\",\"owned_by\":\"breeze-lora-server\"}"
        << "]}";
    return json_response(out.str());
}

HttpResponse ServerRuntime::handle_ui_index() const {
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/html; charset=utf-8";
    const auto html = embedded_ui_html();
    response.body.assign(html.data(), html.size());
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
}

HttpResponse ServerRuntime::handle_ui_voices() const {
    std::ostringstream out;
    out << "{\"management_enabled\":" << (ui_management_enabled() ? "true" : "false")
        << ",\"voices\":[";
    bool first = true;
    {
        std::lock_guard<std::mutex> lock(defaults_mutex_);
        for (const auto & entry : config_.voices) {
            if (!first) out << ',';
            first = false;
            out << "{\"id\":" << json_quote(entry.id)
                << ",\"default_instruction\":" << json_quote(entry.default_instruction)
                << ",\"has_voice_ref\":" << (entry.voice_ref.has_value() ? "true" : "false")
                << ",\"reference_text\":" << json_quote(entry.reference_text)
                << "}";
        }
    }
    out << "]}";
    return json_response(out.str());
}

HttpResponse ServerRuntime::handle_ui_voice_update(
    const HttpRequest & request,
    const std::string & voice_id) {
    if (!ui_management_enabled()) {
        return error_response(403, "management UI requires a loopback bind", "invalid_request_error");
    }

    const auto body = engine::io::json::parse(request.body);
    if (!body.is_object()) {
        return error_response(400, "request body must be a JSON object", "invalid_request_error");
    }
    const auto default_instruction = engine::io::json::require_string(body, "default_instruction");
    if (default_instruction.empty()) {
        return error_response(400, "default_instruction must be non-empty", "invalid_request_error");
    }
    const bool clear_reference = engine::io::json::optional_bool(body, "clear_reference", false);
    const auto * reference_text_value = body.find("reference_text");
    const bool has_reference_text_field = reference_text_value != nullptr;
    const bool reference_text_is_null =
        has_reference_text_field && reference_text_value->is_null();
    const bool reference_text_is_string =
        has_reference_text_field && reference_text_value->is_string();
    if (has_reference_text_field && !reference_text_is_null && !reference_text_is_string) {
        return error_response(400, "reference_text must be a string or null", "invalid_request_error");
    }
    if (clear_reference && reference_text_is_string) {
        return error_response(
            400,
            "clear_reference cannot be combined with a non-null reference_text",
            "invalid_request_error");
    }

    ServerConfig pending;
    {
        std::lock_guard<std::mutex> lock(defaults_mutex_);
        pending = config_;
    }
    VoiceEntry * target = nullptr;
    for (auto & entry : pending.voices) {
        if (entry.id == voice_id) {
            target = &entry;
            break;
        }
    }
    if (target == nullptr) {
        return error_response(400, "unknown voice id: " + voice_id, "invalid_request_error");
    }

    target->default_instruction = default_instruction;
    if (clear_reference) {
        target->voice_ref.reset();
        target->voice_ref_source.clear();
        target->reference_text.clear();
        target->reference_text_file_source.reset();
    } else if (reference_text_is_string) {
        if (!target->voice_ref.has_value()) {
            return error_response(
                400,
                "reference_text requires a stored reference WAV; upload one first",
                "invalid_request_error");
        }
        const auto text = reference_text_value->as_string();
        if (text.empty()) {
            return error_response(400, "reference_text must be non-empty", "invalid_request_error");
        }
        target->reference_text = text;
        target->reference_text_file_source.reset();
    }

    save_config_atomically(pending);

    VoiceDefaults next_defaults;
    next_defaults.default_instruction = target->default_instruction;
    next_defaults.reference_text = target->reference_text;
    {
        std::lock_guard<std::mutex> lock(defaults_mutex_);
        const auto defaults_it = voice_defaults_.find(voice_id);
        if (defaults_it != voice_defaults_.end() && !clear_reference) {
            next_defaults.reference_codes = defaults_it->second.reference_codes;
        }
        if (clear_reference) {
            next_defaults.reference_codes.reset();
        }
        for (auto & entry : config_.voices) {
            if (entry.id == voice_id) {
                entry = *target;
                break;
            }
        }
        voice_defaults_[voice_id] = std::move(next_defaults);
    }

    return json_response("{\"ok\":true}");
}

HttpResponse ServerRuntime::handle_ui_reference_upload(
    const HttpRequest & request,
    const std::string & voice_id) {
    if (!ui_management_enabled()) {
        return error_response(403, "management UI requires a loopback bind", "invalid_request_error");
    }

    const auto content_type = request_content_type(request);
    const auto boundary = extract_multipart_boundary(content_type);
    if (!boundary.has_value()) {
        return error_response(400, "multipart/form-data required", "invalid_request_error");
    }
    const auto parts = parse_multipart_body(request.body, *boundary);
    const auto * audio_part = find_part(parts, "reference_audio");
    const auto * text_part = find_part(parts, "reference_text");
    const auto * instruction_part = find_part(parts, "default_instruction");
    if (audio_part == nullptr || audio_part->data.empty()) {
        return error_response(400, "reference_audio WAV upload is required", "invalid_request_error");
    }
    if (text_part == nullptr || text_part->data.empty()) {
        return error_response(400, "reference_text must be non-empty", "invalid_request_error");
    }
    std::optional<std::string> instruction_override;
    if (instruction_part != nullptr) {
        if (instruction_part->data.empty()) {
            return error_response(
                400,
                "default_instruction must be non-empty when provided",
                "invalid_request_error");
        }
        instruction_override = instruction_part->data;
    }

    engine::runtime::AudioBuffer audio;
    try {
        audio = load_voice_ref_bytes(audio_part->data, "uploaded reference_audio");
    } catch (const std::exception & ex) {
        return error_response(400, ex.what(), "invalid_request_error");
    }

    ServerConfig pending;
    ServerConfig previous;
    {
        std::lock_guard<std::mutex> lock(defaults_mutex_);
        pending = config_;
        previous = config_;
    }
    VoiceEntry * target = nullptr;
    for (auto & entry : pending.voices) {
        if (entry.id == voice_id) {
            target = &entry;
            break;
        }
    }
    if (target == nullptr) {
        return error_response(400, "unknown voice id: " + voice_id, "invalid_request_error");
    }

    const std::string relative = "webui-references/" + voice_id + ".wav";
    const auto absolute = (pending.config_dir / relative).lexically_normal();
    const auto upload_tmp = std::filesystem::path(absolute.string() + ".tmp");
    std::filesystem::create_directories(absolute.parent_path());

    bool config_saved = false;
    try {
        // Stage to a sibling temp file so a failed encode/save never truncates an
        // existing published reference WAV in place.
        write_binary_file(upload_tmp, audio_part->data);
        (void) load_voice_ref(upload_tmp);

        target->voice_ref = absolute;
        target->voice_ref_source = relative;
        target->reference_text = text_part->data;
        target->reference_text_file_source.reset();
        if (instruction_override.has_value()) {
            target->default_instruction = *instruction_override;
        }

        VoiceDefaults next_defaults;
        next_defaults.default_instruction = target->default_instruction;
        next_defaults.reference_text = target->reference_text;
        if (generator_ != nullptr) {
            next_defaults.reference_codes = generator_->encode_reference(audio);
        }

        save_config_atomically(pending);
        config_saved = true;
        replace_file_atomically(upload_tmp, absolute);

        {
            std::lock_guard<std::mutex> lock(defaults_mutex_);
            for (auto & entry : config_.voices) {
                if (entry.id == voice_id) {
                    entry = *target;
                    break;
                }
            }
            voice_defaults_[voice_id] = std::move(next_defaults);
        }
    } catch (const std::exception & ex) {
        std::error_code ec;
        std::filesystem::remove(upload_tmp, ec);
        if (config_saved) {
            try {
                save_config_atomically(previous);
            } catch (...) {
                // Best-effort restore; surface the original failure below.
            }
        }
        {
            std::lock_guard<std::mutex> lock(defaults_mutex_);
            config_ = previous;
            // Restore defaults instruction/reference text from previous config entry.
            for (const auto & entry : previous.voices) {
                if (entry.id != voice_id) {
                    continue;
                }
                auto & defaults = voice_defaults_[voice_id];
                defaults.default_instruction = entry.default_instruction;
                defaults.reference_text = entry.reference_text;
                if (!entry.voice_ref.has_value()) {
                    defaults.reference_codes.reset();
                }
                break;
            }
        }
        return error_response(400, ex.what(), "invalid_request_error");
    }

    return json_response("{\"ok\":true}");
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
    const auto input = engine::io::json::require_string(body, "input");
    if (input.empty()) {
        return error_response(400, "input must be non-empty", "invalid_request_error");
    }

    auto job = std::make_shared<Job>();
    job->text = input;
    job->response_format = parsed_format;

    std::optional<std::string> voice;
    if (const auto * voice_value = body.find("voice"); voice_value != nullptr && !voice_value->is_null()) {
        voice = voice_value->as_string();
    }
    if (const auto selection_error = resolve_voice_selection(model, voice, *job)) {
        return *selection_error;
    }

    VoiceDefaults defaults_copy;
    defaults_copy.default_instruction = kBuiltInDefaultInstruction;
    if (!job->voice.empty()) {
        std::lock_guard<std::mutex> lock(defaults_mutex_);
        const auto defaults_it = voice_defaults_.find(job->voice);
        if (defaults_it == voice_defaults_.end()) {
            return error_response(500, "missing voice defaults for: " + job->voice, "server_error");
        }
        defaults_copy = defaults_it->second;
    }

    const bool has_top_level_instruction = body.find("instruction") != nullptr;
    job->instruction = engine::io::json::optional_string(body, "instruction", defaults_copy.default_instruction);
    if (const auto * options = body.find("options"); options != nullptr && options->is_object()) {
        if (options->find("instruction") != nullptr) {
            job->instruction = engine::io::json::optional_string(*options, "instruction", job->instruction);
        } else if (!has_top_level_instruction) {
            job->instruction = defaults_copy.default_instruction;
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
    } else if (defaults_copy.reference_codes.has_value()) {
        job->reference_codes = defaults_copy.reference_codes;
        job->reference_text = defaults_copy.reference_text;
    }

    job->seed = static_cast<uint64_t>(engine::io::json::optional_i64(body, "seed", 0));
    job->guidance_scale = engine::io::json::optional_f32(body, "guidance_scale", 1.0F);
    job->temperature = engine::io::json::optional_f32(body, "temperature", 0.9F);
    job->depth_temperature = engine::io::json::optional_f32(body, "depth_temperature", 0.9F);
    job->top_k = engine::io::json::optional_i64(body, "top_k", 50);
    job->top_p = engine::io::json::optional_f32(body, "top_p", 1.0F);
    job->max_tokens = engine::io::json::optional_i64(body, "max_tokens", 1500);

    return enqueue_job(std::move(job));
}

HttpResponse ServerRuntime::handle_ui_speech(const HttpRequest & request) {
    const auto content_type = request_content_type(request);
    if (const auto boundary = extract_multipart_boundary(content_type)) {
        const auto parts = parse_multipart_body(request.body, *boundary);
        const auto * model_part = find_part(parts, "model");
        const auto * voice_part = find_part(parts, "voice");
        const auto * input_part = find_part(parts, "input");
        const auto * instruction_part = find_part(parts, "instruction");
        const auto * audio_part = find_part(parts, "reference_audio");
        const auto * text_part = find_part(parts, "reference_text");
        if (model_part == nullptr || model_part->data.empty()) {
            return error_response(400, "model is required", "invalid_request_error");
        }
        if (input_part == nullptr || input_part->data.empty()) {
            return error_response(400, "input must be non-empty", "invalid_request_error");
        }

        const bool has_audio = audio_part != nullptr && !audio_part->data.empty();
        const bool has_text = text_part != nullptr && !text_part->data.empty();
        if (has_audio != has_text) {
            return error_response(
                400,
                "reference_audio and reference_text must be provided together",
                "invalid_request_error");
        }

        auto job = std::make_shared<Job>();
        job->text = input_part->data;
        job->response_format = ResponseFormat::Wav;
        std::optional<std::string> voice;
        if (voice_part != nullptr) {
            voice = voice_part->data;
        }
        if (const auto selection_error = resolve_voice_selection(model_part->data, voice, *job)) {
            return *selection_error;
        }

        VoiceDefaults defaults_copy;
        defaults_copy.default_instruction = kBuiltInDefaultInstruction;
        if (!job->voice.empty()) {
            std::lock_guard<std::mutex> lock(defaults_mutex_);
            const auto defaults_it = voice_defaults_.find(job->voice);
            if (defaults_it == voice_defaults_.end()) {
                return error_response(500, "missing voice defaults for: " + job->voice, "server_error");
            }
            defaults_copy = defaults_it->second;
        }

        if (instruction_part != nullptr && !instruction_part->data.empty()) {
            job->instruction = instruction_part->data;
        } else {
            job->instruction = defaults_copy.default_instruction;
        }
        if (job->instruction.empty()) {
            return error_response(400, "instruction must be non-empty", "invalid_request_error");
        }

        if (has_audio) {
            try {
                job->reference_audio = load_voice_ref_bytes(audio_part->data, "uploaded reference_audio");
            } catch (const std::exception & ex) {
                return error_response(400, ex.what(), "invalid_request_error");
            }
            job->reference_text = text_part->data;
        } else if (defaults_copy.reference_codes.has_value()) {
            job->reference_codes = defaults_copy.reference_codes;
            job->reference_text = defaults_copy.reference_text;
        }

        return enqueue_job(std::move(job));
    }

    return handle_speech(request.body);
}

void ServerRuntime::activate_voice(const std::string & voice_id) {
    if (activation_id_for_voice(voice_id) == kBreezeBaseModelId) {
        generator_->lora_manager().activate_base();
        return;
    }
    generator_->lora_manager().activate_adapter(voice_id);
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

    const std::string activation_id = activation_id_for_voice(job.voice);
    const bool first_load = !activated_once_.count(activation_id);
    const std::string voice_label = job.voice.empty() ? std::string("(none)") : job.voice;
    double load_ms = 0.0;
    double generate_ms = 0.0;

    try {
        if (generator_ == nullptr) {
            throw std::runtime_error("synthesis unavailable in config-only mode");
        }
        if (generator_->lora_manager().active_id() != activation_id) {
            const auto load_started = clock::now();
            activate_voice(job.voice);
            load_ms = ms_since(load_started);
            activated_once_.insert(activation_id);
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
            "speech request model=" + std::string(kPublicModelId)
                + " voice=" + voice_label
                + " first_load=" + (first_load ? "true" : "false")
                + " load_ms=" + format_ms(load_ms)
                + " generate_ms=" + format_ms(generate_ms)
                + " status=ok");
        return response;
    } catch (const std::exception &) {
        engine::debug::log_message(
            engine::debug::LogLevel::Error,
            "breeze_lora_server",
            "speech request model=" + std::string(kPublicModelId)
                + " voice=" + voice_label
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
