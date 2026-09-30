#include "engine/models/breeze_tts/lora.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <vector>

using engine::models::breeze_tts::breeze_lora_linear_forward;
using engine::models::breeze_tts::breeze_lora_merge_weight;
using engine::test::require;

namespace {

bool nearly_equal(float a, float b, float eps = 1e-5F) {
    return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
    constexpr int64_t out_features = 3;
    constexpr int64_t in_features = 4;
    constexpr int64_t rank = 2;
    constexpr int64_t batch = 2;
    const float scale = 2.0F;

    std::vector<float> weight = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    std::vector<float> lora_a = {
        1, 0, 0, 0,
        0, 1, 0, 0,
    };
    std::vector<float> lora_b = {
        1, 0,
        0, 1,
        0, 0,
    };
    std::vector<float> input = {
        1, 2, 3, 4,
        0, 1, 0, 1,
    };

    std::vector<float> disabled;
    breeze_lora_linear_forward(
        weight, lora_a, lora_b, scale, false, out_features, in_features, rank, input, batch, disabled);
    require(disabled.size() == static_cast<size_t>(batch * out_features), "disabled output size");
    require(nearly_equal(disabled[0], 1.0F) && nearly_equal(disabled[1], 2.0F) && nearly_equal(disabled[2], 3.0F),
            "disabled first row");
    require(nearly_equal(disabled[3], 0.0F) && nearly_equal(disabled[4], 1.0F) && nearly_equal(disabled[5], 0.0F),
            "disabled second row");

    std::vector<float> enabled;
    breeze_lora_linear_forward(
        weight, lora_a, lora_b, scale, true, out_features, in_features, rank, input, batch, enabled);
    // mid = x @ A^T = [[1,2],[0,1]]; update = mid @ B^T * scale = [[2,4,0],[0,2,0]]
    require(nearly_equal(enabled[0], 3.0F) && nearly_equal(enabled[1], 6.0F) && nearly_equal(enabled[2], 3.0F),
            "enabled first row");
    require(nearly_equal(enabled[3], 0.0F) && nearly_equal(enabled[4], 3.0F) && nearly_equal(enabled[5], 0.0F),
            "enabled second row");

    const auto merged = breeze_lora_merge_weight(weight, lora_a, lora_b, scale, out_features, in_features, rank);
    std::vector<float> via_merge;
    breeze_lora_linear_forward(
        merged, lora_a, lora_b, scale, false, out_features, in_features, rank, input, batch, via_merge);
    require(via_merge.size() == enabled.size(), "merge output size");
    for (size_t i = 0; i < enabled.size(); ++i) {
        require(nearly_equal(via_merge[i], enabled[i]), "merge equivalence");
    }

    std::cout << "test_breeze_lora_math: ok\n";
    return 0;
}
