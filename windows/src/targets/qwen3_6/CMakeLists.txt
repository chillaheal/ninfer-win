target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/digest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/media_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/resources.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/prefix_identity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/decoder_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/state_image.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/round_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/visual_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/vision_cpu_simd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/control.cpp)

# The AVX-512 kernel TU gets its own arch flag (MSVC: last /arch wins); runtime-dispatched via
# cpu_simd::has_avx512f(), scalar fallback in vision_cpu_impl.h keeps non-AVX-512 hosts correct.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/vision_cpu_simd.cpp
  PROPERTIES COMPILE_FLAGS "/arch:AVX512")

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
