#include "speech_sampling.h"

#include "engine/framework/runtime/options.h"

#include <charconv>
#include <string>

namespace breeze_lora_server {
namespace {

std::optional<std::string> parse_form_i64(std::string_view text, int64_t & out) {
    if (text.empty()) {
        return "must be an integer";
    }
    int64_t value = 0;
    const auto * begin = text.data();
    const auto * end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end) {
        return "must be an integer";
    }
    out = value;
    return std::nullopt;
}

}  // namespace

std::optional<std::string> validate_speech_seed_raw(const int64_t raw) {
    if (raw < kSpeechSeedRandomSentinel || raw > kSpeechSeedMaxU32) {
        return "seed must be -1 or an unsigned 32-bit integer (0 to 4294967295)";
    }
    return std::nullopt;
}

std::optional<std::string> parse_speech_seed_json(
    const engine::io::json::Value & body,
    const int64_t default_when_omitted,
    int64_t & out_raw) {
    const auto * value = body.find("seed");
    if (value == nullptr) {
        out_raw = default_when_omitted;
        return std::nullopt;
    }
    if (!value->is_number()) {
        return "seed must be an integer";
    }
    try {
        out_raw = value->as_i64();
    } catch (...) {
        return "seed must be an integer";
    }
    return validate_speech_seed_raw(out_raw);
}

std::optional<std::string> parse_speech_seed_form(
    const std::string_view text,
    const int64_t default_when_omitted,
    int64_t & out_raw) {
    if (text.empty()) {
        out_raw = default_when_omitted;
        return std::nullopt;
    }
    int64_t parsed = 0;
    if (const auto error = parse_form_i64(text, parsed)) {
        return std::string("seed ") + *error;
    }
    out_raw = parsed;
    return validate_speech_seed_raw(out_raw);
}

uint64_t resolve_speech_seed(const int64_t raw) {
    if (raw == kSpeechSeedRandomSentinel) {
        return static_cast<uint64_t>(engine::runtime::random_u32_seed());
    }
    return static_cast<uint64_t>(raw);
}

}  // namespace breeze_lora_server
