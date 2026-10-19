#pragma once

#include "engine/framework/io/json.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace breeze_lora_server {

inline constexpr int64_t kSpeechSeedRandomSentinel = -1;
inline constexpr int64_t kSpeechSeedMaxU32 = 4294967295LL;

std::optional<std::string> validate_speech_seed_raw(int64_t raw);

std::optional<std::string> parse_speech_seed_json(
    const engine::io::json::Value & body,
    int64_t default_when_omitted,
    int64_t & out_raw);

std::optional<std::string> parse_speech_seed_form(
    std::string_view text,
    int64_t default_when_omitted,
    int64_t & out_raw);

uint64_t resolve_speech_seed(int64_t raw);

}  // namespace breeze_lora_server
