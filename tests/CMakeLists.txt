monowire_add_compile_flags()

find_package(Threads REQUIRED)
find_package(Python3 QUIET COMPONENTS Interpreter)

set(MONOWIRE_TEST_MODELFILE "" CACHE FILEPATH
    "Path to a GGUF model used by extended monowire integration tests")

function(monowire_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED MONOWIRE_TEST_NAME)
        set(TEST_TARGET ${MONOWIRE_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE monowire monowire-common)

    if (MONOWIRE_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(monowire_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME WORKING_DIRECTORY)
    set(multiValueArgs LABEL ARGS)
    cmake_parse_arguments(MONOWIRE_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED MONOWIRE_TEST_LABEL)
        set(MONOWIRE_TEST_LABEL "core")
    endif()
    if (NOT DEFINED MONOWIRE_TEST_WORKING_DIRECTORY)
        set(MONOWIRE_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED MONOWIRE_TEST_NAME)
        set(TEST_NAME ${MONOWIRE_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${MONOWIRE_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${target}> ${MONOWIRE_TEST_ARGS}
    )

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${MONOWIRE_TEST_LABEL})
endfunction()

function(monowire_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME WORKING_DIRECTORY)
    set(multiValueArgs LABEL ARGS)
    cmake_parse_arguments(MONOWIRE_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${MONOWIRE_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (DEFINED MONOWIRE_TEST_NAME)
        set(TEST_TARGET ${MONOWIRE_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    if (NOT DEFINED MONOWIRE_TEST_LABEL)
        set(MONOWIRE_TEST_LABEL "core")
    endif()
    if (NOT DEFINED MONOWIRE_TEST_WORKING_DIRECTORY)
        set(MONOWIRE_TEST_WORKING_DIRECTORY .)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE monowire monowire-common)

    if (MONOWIRE_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${MONOWIRE_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}> ${MONOWIRE_TEST_ARGS}
    )

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${MONOWIRE_TEST_LABEL})
endfunction()

function(monowire_set_model_env test_name)
    if (MONOWIRE_TEST_MODELFILE)
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "MONOWIRE_TEST_MODELFILE=${MONOWIRE_TEST_MODELFILE}")
    endif()
endfunction()

monowire_build(test-tokenizer-0.cpp)

monowire_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge       LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-command-r      LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm   LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-falcon         LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4        LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2          LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-monowire-bpe   LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-monowire-bpe.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-monowire-spm   LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-monowire-spm.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-mpt            LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3          LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2          LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-refact         LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
monowire_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder      LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

monowire_build(test-tokenizer-1-spm.cpp)
monowire_test(test-tokenizer-1-spm NAME test-tokenizer-1-monowire-spm LABEL "tokenizer" ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-monowire-spm.gguf)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    monowire_build_and_test(test-sampling.cpp             LABEL "core")
    monowire_build_and_test(test-reasoning-budget.cpp     LABEL "core")
    monowire_build_and_test(test-grammar-parser.cpp       LABEL "core")
    monowire_build_and_test(test-grammar-integration.cpp  LABEL "core")
    monowire_build_and_test(test-monowire-grammar.cpp        LABEL "core")
    monowire_build_and_test(test-json-schema-to-grammar.cpp LABEL "core;slow")
    monowire_build_and_test(test-monowire-archs.cpp          LABEL "core;slow")

    if (Python3_Interpreter_FOUND)
        set_tests_properties(test-json-schema-to-grammar PROPERTIES
            ENVIRONMENT
                "MONOWIRE_JSON_SCHEMA_TO_GRAMMAR_PY=${PROJECT_SOURCE_DIR}/scripts/json_schema_to_grammar.py;MONOWIRE_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}")
    endif()
endif()

monowire_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp LABEL "core")
monowire_build_and_test(test-jinja.cpp LABEL "core")
if (Python3_Interpreter_FOUND)
    monowire_test(test-jinja NAME test-jinja-python LABEL "python;slow" ARGS -py)
endif()
monowire_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} LABEL "core")
monowire_build_and_test(test-chat-template.cpp    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} LABEL "core;slow")
monowire_build_and_test(test-json-partial.cpp LABEL "core")
monowire_build_and_test(test-log.cpp          LABEL "core")
monowire_build_and_test(test-monowire-streaming.cpp    LABEL "core")
monowire_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    LABEL "core")
monowire_build_and_test(test-regex-partial.cpp LABEL "core")
monowire_build_and_test(test-arg-parser.cpp    LABEL "core")
monowire_build_and_test(test-gguf.cpp          LABEL "core")
monowire_build_and_test(test-backend-ops.cpp   LABEL "backend;slow")

if (NOT GGML_BACKEND_DL)
    monowire_build_and_test(test-barrier.cpp      LABEL "backend")
    monowire_build_and_test(test-quantize-fns.cpp LABEL "backend")
    monowire_build_and_test(test-rope.cpp         LABEL "backend")
endif()

if (NOT MONOWIRE_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
    monowire_build_and_test(test-opt.cpp LABEL "backend;slow")
endif()

set(TEST_TARGET test-c)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE monowire)
if (MONOWIRE_TESTS_INSTALL)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endif()
monowire_test(${TEST_TARGET} LABEL "core")

monowire_build_and_test(test-alloc.cpp LABEL "backend")
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

monowire_build_and_test(test-model-load-cancel.cpp LABEL "model")
monowire_set_model_env(test-model-load-cancel)

monowire_build_and_test(test-autorelease.cpp LABEL "model")
monowire_set_model_env(test-autorelease)

monowire_build_and_test(test-backend-sampler.cpp LABEL "model;slow")
monowire_set_model_env(test-backend-sampler)

if (MONOWIRE_TEST_MODELFILE)
    monowire_build_and_test(
        test-thread-safety.cpp
        LABEL "model;slow"
        ARGS -m "${MONOWIRE_TEST_MODELFILE}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
    monowire_build_and_test(
        test-state-restore-fragmented.cpp
        LABEL "model"
        ARGS -m "${MONOWIRE_TEST_MODELFILE}")
else()
    message(STATUS "MONOWIRE_TEST_MODELFILE is not set; thread-safety/state restore integration tests will not be registered")
endif()

add_custom_target(check-fast
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -LE "model|slow"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    USES_TERMINAL
)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -LE "model"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    USES_TERMINAL
)

add_custom_target(check-model
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L "model"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    USES_TERMINAL
)

add_custom_target(check-all
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    USES_TERMINAL
)
