# Workaround for ABI mismatch between libswoole.so (Debug) and Release builds
# The stack protector detects false positives due to different compilation settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector")

# CRITICAL: libswoole.so is compiled with HAVE_CONFIG_H which includes config.h
# This affects the Server class layout. We MUST match this.
add_compile_definitions(HAVE_CONFIG_H)

# Required for httplib SSL client support
add_compile_definitions(CPPHTTPLIB_OPENSSL_SUPPORT)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# Swoole configuration
set(SWOOLE_SRC_DIR "/home/jason-dev/swoole/swoole-src")
set(SWOOLE_INCLUDE_DIR "${SWOOLE_SRC_DIR}/include")
set(SWOOLE_LIBRARY_DIR "${SWOOLE_SRC_DIR}/lib")

# llama.cpp configuration - use the build directory from parent cmake
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}")
set(LLAMA_BUILD_DIR "${CMAKE_BINARY_DIR}")
set(LLAMA_COMMON_LIB_DIR "${LLAMA_BUILD_DIR}/common")
set(LLAMA_SHARED_LIB_DIR "${LLAMA_BUILD_DIR}/bin")
set(LLAMA_HTTPLIB_LIB_DIR "${LLAMA_BUILD_DIR}/vendor/cpp-httplib")

# server-context containing the core server logic, used by llama-server-coro and CLI

set(TARGET server-context-coro)

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR})
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/common)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/include)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/ggml/include)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_include_directories(${TARGET} PRIVATE ${LLAMA_BUILD_DIR})
target_include_directories(${TARGET} PRIVATE ${SWOOLE_INCLUDE_DIR})
target_include_directories(${TARGET} PRIVATE ${SWOOLE_SRC_DIR})

target_link_directories(${TARGET} PRIVATE ${LLAMA_COMMON_LIB_DIR})
target_link_directories(${TARGET} PRIVATE ${LLAMA_SHARED_LIB_DIR})
# Link to library files directly to avoid inheriting cmake target definitions
target_link_libraries(${TARGET} PRIVATE 
    ${LLAMA_COMMON_LIB_DIR}/libcommon.a
    ${LLAMA_SHARED_LIB_DIR}/libmtmd.so
    ${CMAKE_THREAD_LIBS_INIT})


# llama-server-coro executable (Swoole coroutine-based)

set(TARGET llama-server-coro)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
    httplib_client.cpp
)
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR})
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/common)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/include)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/ggml/include)
target_include_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_include_directories(${TARGET} PRIVATE ${LLAMA_BUILD_DIR})
target_include_directories(${TARGET} PRIVATE ${SWOOLE_INCLUDE_DIR})
target_include_directories(${TARGET} PRIVATE ${SWOOLE_SRC_DIR})

target_link_directories(${TARGET} PRIVATE ${LLAMA_COMMON_LIB_DIR})
target_link_directories(${TARGET} PRIVATE ${LLAMA_SHARED_LIB_DIR})
target_link_directories(${TARGET} PRIVATE ${LLAMA_HTTPLIB_LIB_DIR})
target_link_directories(${TARGET} PRIVATE ${SWOOLE_LIBRARY_DIR})
# Link to library files directly to avoid inheriting cmake target definitions
# We use httplib_coro namespace for our coroutine Server, but libcommon.a and
# server-models.cpp use original httplib::Client from libcpp-httplib.a
target_link_libraries(${TARGET} PRIVATE 
    server-context-coro 
    ${LLAMA_COMMON_LIB_DIR}/libcommon.a
    ${SWOOLE_LIBRARY_DIR}/libswoole.so
    ${LLAMA_SHARED_LIB_DIR}/libmtmd.so
    ${LLAMA_SHARED_LIB_DIR}/libllama.so
    ${LLAMA_SHARED_LIB_DIR}/libggml.so
    ${LLAMA_SHARED_LIB_DIR}/libggml-base.so
    ${LLAMA_SHARED_LIB_DIR}/libggml-cpu.so
    ${LLAMA_HTTPLIB_LIB_DIR}/libcpp-httplib.a
    ssl crypto 
    ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
