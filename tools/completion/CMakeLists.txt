set(TARGET monowire-cli)
add_executable(${TARGET} completion.cpp)
target_link_libraries(${TARGET} PRIVATE monowire-common monowire ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(MONOWIRE_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
