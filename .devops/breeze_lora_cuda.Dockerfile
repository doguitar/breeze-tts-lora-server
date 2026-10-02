# Dedicated Breeze + LoRA hot-swap server image (CUDA).
ARG CUDA_VERSION=12.4.1
ARG CUDA_DOCKER_ARCH=native
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build
ARG CUDA_DOCKER_ARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_ENABLE_CUDA=ON \
      -DENGINE_BUILD_TESTS=ON \
      -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
 && cmake --build build --parallel --target breeze_lora_server breeze_lora_math_test breeze_lora_manifest_test breeze_lora_server_config_test

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/bin/breeze_lora_server /app/breeze_lora_server
COPY --from=build /src/build/bin/breeze_lora_math_test /app/breeze_lora_math_test
COPY --from=build /src/build/bin/breeze_lora_manifest_test /app/breeze_lora_manifest_test
COPY --from=build /src/build/bin/breeze_lora_server_config_test /app/breeze_lora_server_config_test
COPY app/breeze_lora_server/example.server.json /app/example.server.json
EXPOSE 8080
ENTRYPOINT ["/app/breeze_lora_server", "--config", "/app/server.json"]
