#include "speech_sampling.h"
#include "test_assert.h"

#include <limits>

using breeze_lora_server::kSpeechSeedMaxU32;
using breeze_lora_server::kSpeechSeedRandomSentinel;
using breeze_lora_server::parse_speech_seed_form;
using breeze_lora_server::parse_speech_seed_json;
using breeze_lora_server::resolve_speech_seed;
using breeze_lora_server::validate_speech_seed_raw;
using engine::test::require;

namespace {

void test_validate_speech_seed_raw() {
    require(!validate_speech_seed_raw(kSpeechSeedRandomSentinel).has_value(), "accept -1");
    require(!validate_speech_seed_raw(0).has_value(), "accept 0");
    require(!validate_speech_seed_raw(kSpeechSeedMaxU32).has_value(), "accept max u32");
    require(validate_speech_seed_raw(-2).has_value(), "reject -2");
    require(validate_speech_seed_raw(kSpeechSeedMaxU32 + 1).has_value(), "reject above max");
}

void test_parse_speech_seed_json() {
    const auto body = engine::io::json::parse(R"({"seed":42})");
    int64_t raw = 0;
    require(!parse_speech_seed_json(body, 0, raw).has_value(), "valid seed parses");
    require(raw == 42, "parsed value");

    const auto omitted = engine::io::json::parse(R"({"input":"x"})");
    raw = 99;
    require(!parse_speech_seed_json(omitted, 0, raw).has_value(), "omitted uses default");
    require(raw == 0, "default when omitted");

    const auto bad_range = engine::io::json::parse(R"({"seed":4294967296})");
    require(parse_speech_seed_json(bad_range, 0, raw).has_value(), "reject out of range");

    const auto non_int = engine::io::json::parse(R"({"seed":1.5})");
    require(parse_speech_seed_json(non_int, 0, raw).has_value(), "reject non-integer");

    const auto string_seed = engine::io::json::parse(R"({"seed":"abc"})");
    require(parse_speech_seed_json(string_seed, 0, raw).has_value(), "reject string seed");
}

void test_parse_speech_seed_form() {
    int64_t raw = 0;
    require(!parse_speech_seed_form("-1", 0, raw).has_value(), "form -1 ok");
    require(raw == -1, "form -1 value");
    require(!parse_speech_seed_form("", 7, raw).has_value(), "empty uses default");
    require(raw == 7, "empty default");
    require(parse_speech_seed_form("abc", 0, raw).has_value(), "reject non-numeric form");
}

void test_resolve_speech_seed_random() {
    const uint64_t resolved = resolve_speech_seed(kSpeechSeedRandomSentinel);
    require(resolved <= static_cast<uint64_t>(kSpeechSeedMaxU32), "random resolves within u32");
    require(resolved != std::numeric_limits<uint64_t>::max(), "random is not UINT64_MAX");
}

}  // namespace

int main() {
    test_validate_speech_seed_raw();
    test_parse_speech_seed_json();
    test_parse_speech_seed_form();
    test_resolve_speech_seed_random();
    return 0;
}
