cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(WIN32)
  # Sources are UTF-8; keep MSVC from assuming the system code page. Scoped away from CUDA so
  # nvcc never sees a host-only flag.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  # CCCL (the CUDA C++ standard library) hard-errors on MSVC's traditional preprocessor; upstream
  # compiles with GCC/Clang, where the conforming preprocessor is the default. For .cu files the
  # flag must reach cl through nvcc via -Xcompiler.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  # windows.h (pulled in by NVTX on Windows) defines the max/min function-like macros that break
  # std::numeric_limits<...>::max(); NOMINMAX suppresses them.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/DNOMINMAX>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/DNOMINMAX>)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(WIN32)
  # Windows: no pkg-config. FFmpeg and libcurl are prebuilt trees (see deps/); create imported
  # targets with the same names the Linux pkg-config path produces so consumers stay unchanged.
  set(NINFER_FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/../deps/ffmpeg" CACHE PATH
    "Prebuilt FFmpeg root (include/, lib/*.lib, bin/*.dll)")
  if(NOT EXISTS "${NINFER_FFMPEG_ROOT}/include/libavformat/avformat.h")
    message(FATAL_ERROR "FFmpeg headers not found under NINFER_FFMPEG_ROOT=${NINFER_FFMPEG_ROOT}")
  endif()
  # INTERFACE IMPORTED (not UNKNOWN): it carries no library of its own — the real .libs live in
  # INTERFACE_LINK_LIBRARIES — so CMake must not demand an IMPORTED_LOCATION at generate time.
  add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
  set_target_properties(PkgConfig::FFMPEG PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NINFER_FFMPEG_ROOT}/include"
    # One semicolon-separated list: set_target_properties takes strict PROP/VALUE pairs.
    INTERFACE_LINK_LIBRARIES
      "$<LINK_ONLY:${NINFER_FFMPEG_ROOT}/lib/avformat.lib>;$<LINK_ONLY:${NINFER_FFMPEG_ROOT}/lib/avcodec.lib>;$<LINK_ONLY:${NINFER_FFMPEG_ROOT}/lib/avutil.lib>;$<LINK_ONLY:${NINFER_FFMPEG_ROOT}/lib/swscale.lib>")
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    set(NINFER_CURL_ROOT "${CMAKE_SOURCE_DIR}/../deps/curl" CACHE PATH
      "Prebuilt libcurl root (vcpkg installed layout: include/, lib/, share/)")
    list(APPEND CMAKE_PREFIX_PATH "${NINFER_CURL_ROOT}")
    find_package(CURL REQUIRED)
    # vcpkg's CURL::libcurl is itself an ALIAS (of CURL::libcurl_shared); CMake forbids
    # alias-of-alias, so forward through an INTERFACE IMPORTED target instead.
    add_library(PkgConfig::LIBCURL INTERFACE IMPORTED)
    set_target_properties(PkgConfig::LIBCURL PROPERTIES
      INTERFACE_LINK_LIBRARIES "CURL::libcurl_shared")
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
