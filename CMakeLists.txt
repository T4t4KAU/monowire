cmake_minimum_required(VERSION 3.14...3.28)
project("monowire" C CXX)

include(CheckIncludeFileCXX)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MONOWIRE_STANDALONE ON)
    include(git-vars)
else()
    set(MONOWIRE_STANDALONE OFF)
endif()

option(MONOWIRE_USE_SYSTEM_GGML "Use system libggml" OFF)
option(MONOWIRE_WASM_MEM64      "monowire: use 64-bit memory in WASM builds" ON)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    if (MONOWIRE_WASM_MEM64)
        add_compile_options("-sMEMORY64=1")
        add_link_options("-sMEMORY64=1")
    endif()
    add_link_options("-sALLOW_MEMORY_GROWTH=1")
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (MONOWIRE_STANDALONE)
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(MONOWIRE_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(MONOWIRE_TOOLS_INSTALL_DEFAULT ${MONOWIRE_STANDALONE})
endif()

option(MONOWIRE_ALL_WARNINGS           "monowire: enable all compiler warnings"                   ON)
option(MONOWIRE_ALL_WARNINGS_3RD_PARTY "monowire: enable all compiler warnings in 3rd party libs" OFF)
option(MONOWIRE_FATAL_WARNINGS         "monowire: enable -Werror flag"                            OFF)
option(MONOWIRE_SANITIZE_THREAD        "monowire: enable thread sanitizer"                        OFF)
option(MONOWIRE_SANITIZE_ADDRESS       "monowire: enable address sanitizer"                       OFF)
option(MONOWIRE_SANITIZE_UNDEFINED     "monowire: enable undefined sanitizer"                     OFF)
option(MONOWIRE_BUILD_COMMON           "monowire: build common utils library"                     ${MONOWIRE_STANDALONE})
option(MONOWIRE_BUILD_TESTS            "monowire: build tests"                                    ${MONOWIRE_STANDALONE})
option(MONOWIRE_BUILD_TOOLS            "monowire: build inference tools"                          ${MONOWIRE_STANDALONE})
option(MONOWIRE_TOOLS_INSTALL          "monowire: install tools"                                  ${MONOWIRE_TOOLS_INSTALL_DEFAULT})
option(MONOWIRE_TESTS_INSTALL          "monowire: install tests"                                  ON)
option(MONOWIRE_OPENSSL                "monowire: use openssl to support HTTPS"                   ON)
option(MONOWIRE_LLGUIDANCE             "monowire-common: include LLGuidance for structured output" OFF)

option(MONOWIRE_BUILD_CLI      "Build monowire-cli"      ${MONOWIRE_STANDALONE})
option(MONOWIRE_BUILD_BENCH    "Build monowire-bench"    ${MONOWIRE_STANDALONE})
option(MONOWIRE_BUILD_QUANTIZE "Build monowire-quantize" ${MONOWIRE_STANDALONE})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if (NOT DEFINED MONOWIRE_BUILD_NUMBER)
    set(MONOWIRE_BUILD_NUMBER ${BUILD_NUMBER})
endif()
if (NOT DEFINED MONOWIRE_BUILD_COMMIT)
    set(MONOWIRE_BUILD_COMMIT ${BUILD_COMMIT})
endif()
set(MONOWIRE_INSTALL_VERSION 0.0.${MONOWIRE_BUILD_NUMBER})

set(GGML_ALL_WARNINGS   ${MONOWIRE_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${MONOWIRE_FATAL_WARNINGS})

if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

function (monowire_option_depr TYPE OLD)
    if (${OLD})
        set(NEW "${ARGV2}")
        if(NEW)
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead")
            set(${NEW} ON PARENT_SCOPE)
        else()
            message(${TYPE} "${OLD} is deprecated and will be ignored")
        endif()
    endif()
endfunction()

monowire_option_depr(FATAL_ERROR MONOWIRE_CUBLAS              GGML_CUDA)
monowire_option_depr(WARNING     MONOWIRE_CUDA                GGML_CUDA)
monowire_option_depr(WARNING     MONOWIRE_METAL               GGML_METAL)
monowire_option_depr(WARNING     MONOWIRE_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
monowire_option_depr(WARNING     MONOWIRE_NATIVE              GGML_NATIVE)
monowire_option_depr(WARNING     MONOWIRE_RPC                 GGML_RPC)
monowire_option_depr(WARNING     MONOWIRE_SYCL                GGML_SYCL)
monowire_option_depr(WARNING     MONOWIRE_SYCL_F16            GGML_SYCL_F16)
monowire_option_depr(WARNING     MONOWIRE_CANN                GGML_CANN)
monowire_option_depr(WARNING     MONOWIRE_CURL)

include("cmake/license.cmake")
license_add_file("monowire" "LICENSE")

if (MONOWIRE_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT MONOWIRE_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${MONOWIRE_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${MONOWIRE_BUILD_COMMIT})
    add_subdirectory(ggml)
endif()

add_subdirectory(src)

if (MONOWIRE_BUILD_COMMON)
    add_subdirectory(common)
    add_subdirectory(vendor/cpp-httplib)
endif()

if (MONOWIRE_BUILD_COMMON AND MONOWIRE_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

if (MONOWIRE_BUILD_COMMON AND MONOWIRE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if (MONOWIRE_BUILD_COMMON)
    license_generate(monowire-common)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(MONOWIRE_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header files")
set(MONOWIRE_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(MONOWIRE_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary files")

set(MONOWIRE_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/monowire.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/monowire-cpp.h)

set_target_properties(monowire PROPERTIES PUBLIC_HEADER "${MONOWIRE_PUBLIC_HEADERS}")

install(TARGETS monowire LIBRARY PUBLIC_HEADER)

if (MONOWIRE_BUILD_COMMON)
    install(TARGETS monowire-common LIBRARY)
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/monowire-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/monowire-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/monowire
    PATH_VARS MONOWIRE_INCLUDE_INSTALL_DIR
              MONOWIRE_LIB_INSTALL_DIR
              MONOWIRE_BIN_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/monowire-version.cmake
    VERSION ${MONOWIRE_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monowire-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/monowire-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/monowire)

configure_file(cmake/monowire.pc.in
    "${CMAKE_CURRENT_BINARY_DIR}/monowire.pc"
    @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/monowire.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
