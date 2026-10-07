# Vendored libmp3lame 3.100 (encoder-only), scoped for breeze_lora_server.

set(TARGET audiocpp_libmp3lame)

set(LAME_SOURCES
    libmp3lame/VbrTag.c
    libmp3lame/bitstream.c
    libmp3lame/encoder.c
    libmp3lame/fft.c
    libmp3lame/gain_analysis.c
    libmp3lame/id3tag.c
    libmp3lame/lame.c
    libmp3lame/mpglib_interface.c
    libmp3lame/newmdct.c
    libmp3lame/presets.c
    libmp3lame/psymodel.c
    libmp3lame/quantize.c
    libmp3lame/quantize_pvt.c
    libmp3lame/reservoir.c
    libmp3lame/set_get.c
    libmp3lame/tables.c
    libmp3lame/takehiro.c
    libmp3lame/util.c
    libmp3lame/vbrquantize.c
    libmp3lame/version.c
)

add_library(${TARGET} STATIC ${LAME_SOURCES})
target_include_directories(${TARGET}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/libmp3lame"
)
target_compile_definitions(${TARGET} PRIVATE HAVE_CONFIG_H)
target_compile_features(${TARGET} PUBLIC c_std_99)

if (MSVC)
    target_compile_options(${TARGET} PRIVATE /w)
else()
    target_compile_options(${TARGET} PRIVATE -w)
endif()

# LAME uses math library symbols on non-Windows toolchains.
if (NOT WIN32)
    target_link_libraries(${TARGET} PUBLIC m)
endif()
