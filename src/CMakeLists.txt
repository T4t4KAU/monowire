monowire_add_compile_flags()

#
# libraries
#

# monowire

file(GLOB MONOWIRE_MODELS_SOURCES "models/*.cpp")

add_library(monowire
            ../include/monowire.h
            ../include/monowire-cpp.h
            monowire.cpp
            monowire-adapter.cpp
            monowire-arch.cpp
            monowire-batch.cpp
            monowire-chat.cpp
            monowire-context.cpp
            monowire-cparams.cpp
            monowire-grammar.cpp
            monowire-graph.cpp
            monowire-hparams.cpp
            monowire-impl.cpp
            monowire-io.cpp
            monowire-streaming.cpp
            monowire-kv-cache.cpp
            monowire-kv-cache-iswa.cpp
            monowire-memory.cpp
            monowire-memory-hybrid.cpp
            monowire-memory-hybrid-iswa.cpp
            monowire-memory-recurrent.cpp
            monowire-mmap.cpp
            monowire-model-loader.cpp
            monowire-model-saver.cpp
            monowire-model.cpp
            monowire-quant.cpp
            monowire-sampler.cpp
            monowire-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${MONOWIRE_MODELS_SOURCES}
            )

set_target_properties(monowire PROPERTIES
    VERSION ${MONOWIRE_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(monowire PRIVATE .)
target_include_directories(monowire PUBLIC ../include)
target_compile_features   (monowire PRIVATE cxx_std_17) # don't bump

target_link_libraries(monowire PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(monowire PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(monowire PRIVATE MONOWIRE_BUILD)
    target_compile_definitions(monowire PUBLIC  MONOWIRE_SHARED)
endif()
