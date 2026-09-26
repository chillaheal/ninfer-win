# qwen3_8_flash_next target package (P2: identity, residency binder, startup
# planner; P7: QSA v1 ops; P8: NVFP4 sparse MoE kernels + winner-only dispatch
# -- the program arrives with P9+).
target_sources(ninfer_engine PRIVATE
  impl/planner.cpp
  impl/binder.cpp
  impl/package.cpp
  paging/pager.cpp
  hc/gated_residual.cu
  ple/ngram_embedding.cpp
  qsa/qsa_indexer.cu
  qsa/qsa_indexer_k_append.cu
  qsa/qsa_topk.cu
  qsa/qsa_sparse_gqa.cu
  moe/moe_nvfp4.cu
  moe/moe_nvfp4_dispatch.cpp
  impl/runtime/frontend.cpp
  impl/runtime/sequence_planner.cpp
  impl/runtime/pressure_planning.cpp
  impl/runtime/program.cpp
  impl/runtime/real_program.cpp
  impl/runtime/cpu_moe.cpp
  impl/runtime/leaves/mini_attn.cu
  impl/runtime/leaves/mini_layout.cu
  impl/runtime/leaves/mini_linear.cu
  impl/runtime/leaves/real_attn.cu
  impl/runtime/leaves/real_layout.cu
  impl/runtime/leaves/real_linear.cu
  impl/load/bindings.cpp
  impl/load/real_bindings.cpp
  impl/load/real_loader.cpp)
# P3.2 CPU MoE: host-only TU, AVX2 FMA dots (the rest of the target stays at
# the x64 baseline; the code falls back to scalar fmaf without __AVX2__).
if(MSVC)
  set_source_files_properties(impl/runtime/cpu_moe.cpp PROPERTIES
    COMPILE_FLAGS "/arch:AVX2")
endif()
target_include_directories(ninfer_engine PRIVATE
  impl
  export)
# Overlay layout: this target can also be attached to the core ninfer-win tree
# via add_subdirectory(); expose the flash-next include root and the feature
# macro so shared core TUs (registry.cpp, engine.cpp) can compile against it.
target_compile_definitions(ninfer_engine PUBLIC NINFER_WITH_FLASH_NEXT=1)
target_include_directories(ninfer_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
