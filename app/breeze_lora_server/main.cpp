#include "config.h"
#include "runtime.h"

#include "http.h"

#include "engine/framework/debug/trace.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
std::atomic<bool> g_stop{false};

bool shutdown_requested() {
    return g_stop.load();
}

void on_signal(int) {
    g_stop.store(true);
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        std::string config_path = "server.json";
        std::optional<std::string> log_file;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
                config_path = argv[++i];
            } else if (arg == "--log-file" && i + 1 < argc) {
                log_file = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                std::cout
                    << "Usage: breeze_lora_server --config server.json [--log-file path]\n"
                    << "  Request logs (model/first_load/load_ms/generate_ms) go to stdout by default.\n";
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        // Dedicated server: request timing logs are on by default (stdout, or --log-file).
        engine::debug::configure_logging(engine::debug::LoggingConfig{
            true,
            log_file,
        });

        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        auto config = breeze_lora_server::load_config(config_path);
        const std::string host = config.host;
        const int port = config.port;
        breeze_lora_server::ServerRuntime runtime(std::move(config));
        std::cout << "breeze_lora_server listening on " << host << ':' << port << std::endl;
        minitts::server::serve_http(host, port, runtime, shutdown_requested, 32ull << 20);
        runtime.request_shutdown();
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "breeze_lora_server failed: " << ex.what() << std::endl;
        return 1;
    }
}
