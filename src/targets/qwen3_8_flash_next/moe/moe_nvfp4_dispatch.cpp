// P8 target dispatch — NVFP4 sparse MoE, winner route only.
//
// The public Op (ops::sparse_moe_nvfp4, include/ninfer/ops/sparse_moe_nvfp4.h)
// keeps BOTH the W4A4 and W4A16 routes so the P8 test can oracle-qualify each
// against the FP64 reference. This target-private dispatch pins the microbench
// winner (out/flash_next_dev/P8.log, decode T=1: W4A16 1418.99 vs W4A4 1424.14
// us/iter) and drops the route argument, so the target's sparse-MoE modeling
// function (P9) takes exactly one route — the measured winner — without
// re-deciding it. If a future re-measure flips the winner, this is the single
// place to update.

#include <ninfer/ops/sparse_moe_nvfp4.h>

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::moe {

// The target's only sparse-MoE entry: the W4A16 route (the P8 winner). Same
// shape as ops::sparse_moe_nvfp4 minus the route parameter.
void moe_nvfp4(const Tensor& x, const ops::SparseMoeNvfp4Geometry& geo,
               const ops::SparseMoeNvfp4Weights& w,
               ops::SparseMoeNvfp4Epilogue epilogue, const Tensor* residual,
               Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream) {
    ops::sparse_moe_nvfp4(x, geo, w, ops::SparseMoeNvfp4Route::W4A16, epilogue, residual,
                          destination, workspace, stream);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::moe
