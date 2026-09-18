#pragma once
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

namespace ninfer::ops::detail {
void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream);

void ngram_candidate_inject_launch(const std::int32_t* ngram_tokens, const std::int32_t* ngram_counts,
                                    std::int32_t* candidate_ids, float* unary, int steps,
                                    int batch, int ngram_concurrency, float boost,
                                    cudaStream_t stream);
}
