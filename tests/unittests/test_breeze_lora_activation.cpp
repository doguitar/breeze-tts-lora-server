#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/models/breeze_tts/lora.h"
#include "test_assert.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using engine::models::breeze_tts::BreezeLoraAdapterTensors;
using engine::models::breeze_tts::BreezeLoraManager;
using engine::models::breeze_tts::BreezeLoraTargetBinding;
using engine::models::breeze_tts::kBreezeBaseModelId;
using engine::models::breeze_tts::kBreezeLoraRank;
using engine::test::require;

namespace {

template <class Fn>
void rejects(Fn fn, const char * message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, message);
}

// Shared fixture geometry. register_adapter validates A/B against the bound target
// (rank*in_features and out_features*rank), so adapters and targets must agree.
constexpr int64_t kFixtureIn = 8;
constexpr int64_t kFixtureOut = 4;

BreezeLoraAdapterTensors make_adapter(float fill_a, float fill_b) {
    BreezeLoraAdapterTensors adapter;
    adapter.lora_a.emplace("m", std::vector<float>(kBreezeLoraRank * kFixtureIn, fill_a));
    adapter.lora_b.emplace("m", std::vector<float>(kFixtureOut * kBreezeLoraRank, fill_b));
    adapter.scale = 2.0F;
    // register_adapter revalidates the manifest, so the fixture must satisfy it.
    adapter.manifest.base_revision = "abc123";
    adapter.manifest.base_files.emplace("config.json", std::string(64, 'a'));
    adapter.manifest.adapter_sha256 = std::string(64, 'b');
    return adapter;
}

// A bound target whose `live` pointer is a real backend tensor, so the test can read
// back the bytes a merge-and-upload implementation would have written.
struct LiveTargetFixture {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ggml = nullptr;
    ggml_tensor * live = nullptr;
    std::array<float, 32> base{};
    BreezeLoraManager manager;

    LiveTargetFixture() {
        backend = engine::core::init_backend(backend_config);
        ggml_init_params params{};
        params.mem_size = 4 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml = ggml_init(params);
        if (backend == nullptr || ggml == nullptr) {
            throw std::runtime_error("failed to initialize base-buffer test backend");
        }
        live = ggml_new_tensor_2d(ggml, GGML_TYPE_F32, 8, 4);
        if (live == nullptr) {
            throw std::runtime_error("failed to create base-buffer test tensor");
        }
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate base-buffer test tensor");
        }
        for (size_t i = 0; i < base.size(); ++i) {
            base[i] = static_cast<float>(i) * 0.25F;
        }
    }

    ~LiveTargetFixture() {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ggml != nullptr) ggml_free(ggml);
        if (backend != nullptr) ggml_backend_free(backend);
    }

    void bind() {
        BreezeLoraTargetBinding binding;
        binding.module_name = "m";
        binding.out_features = 4;
        binding.in_features = 8;
        binding.base_f32.assign(base.begin(), base.end());
        binding.live = live;
        binding.live_shape = engine::core::TensorShape::from_dims({4, 8});
        binding.live_type = GGML_TYPE_F32;
        binding.row_offset = 0;
        manager.bind_target(std::move(binding));
    }

    std::vector<float> read_live() const {
        std::vector<float> out(base.size());
        ggml_backend_tensor_get(live, out.data(), 0, out.size() * sizeof(float));
        return out;
    }
};

// Proves the load-bearing invariant end to end at the manager level: after the real
// upload path runs, switching base -> adapter-a -> adapter-b -> adapter-a -> base must
// leave both the resident backend tensor and the captured base rows byte-identical.
// A merge-and-upload implementation fails this; a side-branch implementation passes.
void verify_base_rows_immutable() {
    LiveTargetFixture fixture;
    fixture.bind();
    auto & manager = fixture.manager;
    ggml_tensor * const key = fixture.live;

    const auto expect_pristine = [&](const char * phase) {
        const auto & rows = manager.base_rows(key);
        require(rows.size() == fixture.base.size(), std::string("base row count at ") + phase);
        const auto live_rows = fixture.read_live();
        require(live_rows.size() == fixture.base.size(), std::string("live row count at ") + phase);
        for (size_t i = 0; i < fixture.base.size(); ++i) {
            require(rows[i] == fixture.base[i], std::string("captured base row at ") + phase);
            require(
                live_rows[i] == fixture.base[i],
                std::string("resident base tensor row at ") + phase);
        }
    };

    manager.upload_base_live_buffers();
    manager.register_adapter("adapter-a", make_adapter(1.0F, 2.0F));
    manager.register_adapter("adapter-b", make_adapter(3.0F, 4.0F));
    expect_pristine("startup");

    manager.activate_adapter("adapter-a");
    expect_pristine("adapter-a");
    manager.activate_adapter("adapter-b");
    expect_pristine("adapter-b");
    manager.activate_adapter("adapter-a");
    expect_pristine("adapter-a-reactivated");
    manager.activate_base();
    expect_pristine("base-restored");
}

}  // namespace

int main() {
    BreezeLoraManager manager;
    manager.register_adapter("adapter-a", make_adapter(1.0F, 2.0F));
    manager.register_adapter("adapter-b", make_adapter(3.0F, 4.0F));

    require(manager.active_id() == kBreezeBaseModelId, "base active at startup");
    require(manager.model_ids().size() == 3, "model id list");
    require(manager.model_ids()[0] == kBreezeBaseModelId, "base listed first");

    const size_t base_count = manager.activation_count();
    manager.activate_base();
    require(manager.activation_count() == base_count, "reselecting base is a no-op");

    manager.activate_adapter("adapter-a");
    require(manager.active_id() == "adapter-a", "adapter-a active");
    require(manager.activation_count() == base_count + 1, "adapter-a activation counted");
    require(manager.active_adapter() != nullptr && manager.active_adapter()->scale == 2.0F, "adapter-a scale");
    require(manager.active_lora_a("m").size() == static_cast<size_t>(kBreezeLoraRank * kFixtureIn), "adapter-a A size");
    require(manager.active_lora_b("m").size() == static_cast<size_t>(kFixtureOut * kBreezeLoraRank), "adapter-a B size");

    const size_t after_a = manager.activation_count();
    manager.activate_adapter("adapter-a");
    require(manager.activation_count() == after_a, "reselecting active adapter is a no-op");

    manager.activate_adapter("adapter-b");
    require(manager.active_id() == "adapter-b", "adapter-b active");
    require(manager.activation_count() == after_a + 1, "adapter-b activation counted");
    require(manager.active_lora_a("m")[0] == 3.0F, "adapter-b A values active");

    manager.activate_adapter("adapter-a");
    require(manager.active_id() == "adapter-a", "adapter-a re-activated");
    require(manager.active_lora_a("m")[0] == 1.0F, "adapter-a A values restored");

    manager.activate_base();
    require(manager.active_id() == kBreezeBaseModelId, "base restored");
    require(manager.active_adapter() == nullptr, "no active adapter on base");
    rejects([&] { manager.active_lora_a("m"); }, "base exposes no A tensors");

    require(manager.activation_count() == after_a + 3, "expected activation count");

    rejects([&] { manager.activate_adapter("missing"); }, "unknown adapter accepted");
    rejects([&] { manager.register_adapter("adapter-a", make_adapter(1.0F, 1.0F)); },
            "duplicate adapter id accepted");
    rejects([&] { manager.register_adapter(kBreezeBaseModelId, make_adapter(1.0F, 1.0F)); },
            "base id accepted as adapter");

    // The load-bearing invariant: activation must not rewrite the resident base rows.
    verify_base_rows_immutable();

    std::cout << "test_breeze_lora_activation: ok\n";
    return 0;
}
