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
  impl/runtime/leaves/mini_attn.cu
  impl/runtime/leaves/mini_layout.cu
  impl/runtime/leaves/mini_linear.cu
  impl/runtime/leaves/real_attn.cu
  impl/runtime/leaves/real_layout.cu
  impl/runtime/leaves/real_linear.cu
  impl/load/bindings.cpp
  impl/load/real_bindings.cpp
  impl/load/real_loader.cpp)
target_include_directories(ninfer_engine PRIVATE
  impl
  export)
